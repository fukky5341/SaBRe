## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 6.28359642836
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.0230141, 2.8226054, -4.0230141, 2.8226054, -6.8456192, 6.8456192)
1: (-2.7556953, 3.0856514, -2.7556953, 3.0856514, -5.8413467, 5.8413467)
2: (-3.9878938, 3.0745039, -3.9878938, 3.0745039, -7.0623980, 7.0623980)
3: (-4.8247290, 2.4369347, -4.8247290, 2.4369347, -7.2616634, 7.2616634)
4: (-5.0468745, 3.2078700, -5.0468745, 3.2078700, -8.2547445, 8.2547445)
5: (-4.2217145, 2.6404290, -4.2217145, 2.6404290, -6.8621435, 6.8621435)
6: (-4.8847556, 2.9497087, -4.8847556, 2.9497087, -7.8344641, 7.8344641)
7: (-3.6985793, 3.6995137, -3.6985793, 3.6995137, -7.3980932, 7.3980932)
8: (-5.3712187, 2.7974010, -5.3712187, 2.7974010, -8.1686192, 8.1686192)
9: (-3.5766788, 3.6042769, -3.5766788, 3.6042769, -7.1809559, 7.1809559)

## BASE Result
execution time: IAR + LP analysis = 1.60 + 4.20 = 5.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -7.1077352, upper bound: 7.1077352


# Binary Search by BASE starts (time budget: 2694.20 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=7.834464073181152
rel_dist={6: [-7.107378872222906, 7.107378872222906]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=7.834464073181152
rel_dist={6: [-7.107152351905672, 7.10715235190567]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=7.834464073181152
rel_dist={6: [-7.106923203393791, 7.106923203247483]}

## Binary Search Result
Binary search time: 25.80 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2668.41 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 3.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055988
time: 2.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.13
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.13
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055988

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.3117657, 2.3313293, -4.0164604, 2.8180699, -6.1298356, 6.3477898
1: -2.2746105, 2.5741804, -2.7508945, 3.0809798, -5.3555903, 5.3250751
2: -3.2746129, 2.6151695, -3.9813323, 3.0702043, -6.3448172, 6.5965018
3: -3.9685380, 2.0463929, -4.8168044, 2.4333367, -6.4018745, 6.8631973
4: -4.1542244, 2.7427235, -5.0387135, 3.2036214, -7.3578458, 7.7814369
5: -3.4665380, 2.2262321, -4.2148066, 2.6365643, -6.1031022, 6.4410386
6: -3.9971840, 2.5632598, -4.8766279, 2.9460826, -6.9432669, 7.4398880
7: -3.0885596, 3.0902281, -3.6929641, 3.6938686, -6.7824283, 6.7831922
8: -4.4174242, 2.3710141, -5.3624730, 2.7934628, -7.2108870, 7.7334871
9: -2.9865417, 3.0139289, -3.5712790, 3.5987482, -6.5852900, 6.5852079

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.0443048, 4.2405229, -3.9695055, 2.7854352, -8.8297405, 8.2100286
1: -4.2253499, 4.5275450, -2.7164278, 3.0474524, -7.2728024, 7.2439728
2: -5.9989243, 4.4117823, -3.9344950, 3.0393255, -9.0382500, 8.3462772
3: -7.2230110, 3.5636163, -4.7600155, 2.4076657, -9.6306763, 8.3236313
4: -7.5507116, 4.5344086, -4.9800467, 3.1730349, -10.7237463, 9.5144558
5: -6.3241363, 3.8430071, -4.1652212, 2.6088462, -8.9329824, 8.0082283
6: -7.4014902, 4.1482449, -4.8183041, 2.9197609, -10.3212509, 8.9665489
7: -5.4129872, 5.4127798, -3.6527381, 3.6534359, -9.0664234, 9.0655174
8: -8.0652170, 4.0733213, -5.2997656, 2.7649760, -10.8301926, 9.3730869
9: -5.2316179, 5.2847004, -3.5325308, 3.5594065, -8.7910242, 8.8172312

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.91 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.91
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.91
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.91
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.91
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -3.3117657, 2.3313293, -3.3117657, 2.3313293, -5.6430950, 5.6430950
1: -2.2746105, 2.5741804, -2.2746105, 2.5741804, -4.8487911, 4.8487911
2: -3.2746129, 2.6151695, -3.2746129, 2.6151695, -5.8897824, 5.8897824
3: -3.9685380, 2.0463929, -3.9685380, 2.0463929, -6.0149307, 6.0149307
4: -4.1542244, 2.7427235, -4.1542244, 2.7427235, -6.8969479, 6.8969479
5: -3.4665380, 2.2262321, -3.4665380, 2.2262321, -5.6927700, 5.6927700
6: -3.9971840, 2.5632598, -3.9971840, 2.5632598, -6.5604439, 6.5604439
7: -3.0885596, 3.0902281, -3.0885596, 3.0902281, -6.1787877, 6.1787877
8: -4.4174242, 2.3710141, -4.4174242, 2.3710141, -6.7884383, 6.7884383
9: -2.9865417, 3.0139289, -2.9865417, 3.0139289, -6.0004706, 6.0004706

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 2.98 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 2.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.3117657, 2.3313293, -6.0443048, 4.2405229, -7.5522885, 8.3756342
1: -2.2746105, 2.5741804, -4.2253499, 4.5275450, -6.8021555, 6.7995300
2: -3.2746129, 2.6151695, -5.9989243, 4.4117823, -7.6863952, 8.6140938
3: -3.9685380, 2.0463929, -7.2230110, 3.5636163, -7.5321541, 9.2694035
4: -4.1542244, 2.7427235, -7.5507116, 4.5344086, -8.6886330, 10.2934351
5: -3.4665380, 2.2262321, -6.3241363, 3.8430071, -7.3095450, 8.5503683
6: -3.9971840, 2.5632598, -7.4014902, 4.1482449, -8.1454287, 9.9647503
7: -3.0885596, 3.0902281, -5.4129872, 5.4127798, -8.5013390, 8.5032158
8: -4.4174242, 2.3710141, -8.0652170, 4.0733213, -8.4907455, 10.4362316
9: -2.9865417, 3.0139289, -5.2316179, 5.2847004, -8.2712421, 8.2455463

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 2.10 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 3.25 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.0443048, 4.2405229, -3.3117657, 2.3313293, -8.3756342, 7.5522885
1: -4.2253499, 4.5275450, -2.2746105, 2.5741804, -6.7995300, 6.8021555
2: -5.9989243, 4.4117823, -3.2746129, 2.6151695, -8.6140938, 7.6863952
3: -7.2230110, 3.5636163, -3.9685380, 2.0463929, -9.2694035, 7.5321541
4: -7.5507116, 4.5344086, -4.1542244, 2.7427235, -10.2934351, 8.6886330
5: -6.3241363, 3.8430071, -3.4665380, 2.2262321, -8.5503683, 7.3095450
6: -7.4014902, 4.1482449, -3.9971840, 2.5632598, -9.9647503, 8.1454287
7: -5.4129872, 5.4127798, -3.0885596, 3.0902281, -8.5032158, 8.5013390
8: -8.0652170, 4.0733213, -4.4174242, 2.3710141, -10.4362316, 8.4907455
9: -5.2316179, 5.2847004, -2.9865417, 3.0139289, -8.2455463, 8.2712421

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055988
time: 2.97 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.17 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.0443048, 4.2405229, -6.0371675, 4.2394943, -10.2837992, 10.2776909
1: -4.2253499, 4.5275450, -4.2221065, 4.5275450, -8.7528954, 8.7496510
2: -5.9989243, 4.4117823, -5.9841914, 4.4115863, -10.4105110, 10.3959732
3: -7.2230110, 3.5636163, -7.2069502, 3.5579624, -10.7809734, 10.7705669
4: -7.5507116, 4.5344086, -7.5488191, 4.5344086, -12.0851202, 12.0832272
5: -6.3241363, 3.8430071, -6.3241363, 3.8381109, -10.1622467, 10.1671429
6: -7.4014902, 4.1482449, -7.4014902, 4.1427288, -11.5442190, 11.5497351
7: -5.4129872, 5.4127798, -5.4059429, 5.4098940, -10.8228817, 10.8187227
8: -8.0652170, 4.0733213, -8.0609074, 4.0716205, -12.1368370, 12.1342287
9: -5.2316179, 5.2847004, -5.2316179, 5.2673373, -10.4989548, 10.5163183

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055988
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.72
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.4147537, 1.0888404, -3.1891837, 2.2416718, -3.6564255, 4.2780242
1: -1.1065094, 1.1250544, -2.1978397, 2.4802232, -3.5867326, 3.3228941
2: -1.2957649, 1.3731751, -3.1455827, 2.5333614, -3.8291264, 4.5187578
3: -1.5482196, 0.9737427, -3.8127713, 1.9754196, -3.5236392, 4.7865143
4: -1.6589015, 1.4458973, -3.9896832, 2.6589241, -4.3178253, 5.4355803
5: -1.4666899, 1.1632832, -3.3277655, 2.1563859, -3.6230760, 4.4910488
6: -1.4416943, 1.5811431, -3.8340845, 2.4945531, -3.9362473, 5.4152279
7: -1.3833644, 1.4175160, -2.9781172, 2.9805717, -4.3639364, 4.3956332
8: -1.7159976, 1.2555535, -4.2425632, 2.2938883, -4.0098858, 5.4981165
9: -1.3292929, 1.3916898, -2.8784156, 2.9075429, -4.2368360, 4.2701054

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
time: 3.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
time: 3.13 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -2.7531009, 1.9244990, -3.2854669, 2.3121364, -5.0652370, 5.2099657
1: -1.9279807, 2.1453786, -2.2581439, 2.5540643, -4.4820452, 4.4035225
2: -2.6867716, 2.2426157, -3.2469697, 2.5976362, -5.2844076, 5.4895854
3: -3.2610307, 1.7218046, -3.9351804, 2.0311718, -5.2922025, 5.6569853
4: -3.4038484, 2.3624566, -4.1189842, 2.7247391, -6.1285877, 6.4814405
5: -2.8362594, 1.9072357, -3.4368219, 2.2112696, -5.0475292, 5.3440576
6: -3.2555003, 2.2518134, -3.9622426, 2.5485334, -5.8040338, 6.2140560
7: -2.5857768, 2.5900977, -3.0648944, 3.0667505, -5.6525273, 5.6549921
8: -3.6204588, 2.0194774, -4.3799515, 2.3544874, -5.9749460, 6.3994288
9: -2.4945264, 2.5285211, -2.9633648, 2.9911404, -5.4856668, 5.4918861

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
time: 3.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
time: 3.06 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.4147537, 1.0888404, -5.9111185, 4.1483245, -5.5630779, 6.9999590
1: -1.1065094, 1.1250544, -4.1278706, 4.4328451, -5.5393543, 5.2529249
2: -1.2957649, 1.3731751, -5.8659859, 4.3244104, -5.6201754, 7.2391610
3: -1.5482196, 0.9737427, -7.0626225, 3.4906299, -5.0388494, 8.0363655
4: -1.6589015, 1.4458973, -7.3848801, 4.4480991, -6.1070004, 8.8307772
5: -1.4666899, 1.1632832, -6.1838140, 3.7643719, -5.2310619, 7.3470974
6: -1.4416943, 1.5811431, -7.2354507, 4.0734911, -5.5151854, 8.8165941
7: -1.3833644, 1.4175160, -5.2993498, 5.2982559, -6.6816206, 6.7168655
8: -1.7159976, 1.2555535, -7.8871260, 3.9921005, -5.7080979, 9.1426792
9: -1.3292929, 1.3916898, -5.1222954, 5.1726770, -6.5019698, 6.5139852

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 4.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 3.92 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.7531009, 1.9244990, -6.0157566, 4.2207742, -6.9738750, 7.9402556
1: -1.9279807, 2.1453786, -4.2044692, 4.5072556, -6.4352360, 6.3498478
2: -2.6867716, 2.2426157, -5.9704409, 4.3930674, -7.0798388, 8.2130566
3: -3.2610307, 1.7218046, -7.1886230, 3.5479763, -6.8090067, 8.9104271
4: -3.4038484, 2.3624566, -7.5151892, 4.5158815, -7.9197302, 9.8776455
5: -2.8362594, 1.9072357, -6.2941003, 3.8261437, -6.6624031, 8.2013359
6: -3.2555003, 2.2518134, -7.3658819, 4.1321869, -7.3876872, 9.6176949
7: -2.5857768, 2.5900977, -5.3886423, 5.3882532, -7.9740300, 7.9787397
8: -3.6204588, 2.0194774, -8.0270767, 4.0559015, -7.6763601, 10.0465546
9: -2.4945264, 2.5285211, -5.2081914, 5.2606959, -7.7552223, 7.7367125

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
time: 2.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.0674763, 2.8777096, -3.1891837, 2.2416718, -6.3091478, 6.0668936
1: -2.7773356, 3.1218820, -2.1978397, 2.4802232, -5.2575588, 5.3197217
2: -4.0264611, 3.1180172, -3.1455827, 2.5333614, -6.5598226, 6.2635999
3: -4.8471227, 2.4828634, -3.8127713, 1.9754196, -6.8225422, 6.2956347
4: -5.0759230, 3.2510345, -3.9896832, 2.6589241, -7.7348471, 7.2407179
5: -4.2390509, 2.6797059, -3.3277655, 2.1563859, -6.3954368, 6.0074711
6: -4.9378076, 3.0720673, -3.8340845, 2.4945531, -7.4323606, 6.9061518
7: -3.7234621, 3.7077055, -2.9781172, 2.9805717, -6.7040339, 6.6858225
8: -5.4250007, 2.9023461, -4.2425632, 2.2938883, -7.7188892, 7.1449094
9: -3.6089909, 3.6207800, -2.8784156, 2.9075429, -6.5165339, 6.4991956

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066073
time: 2.11 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066072
time: 2.10 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.4313140, 3.8164265, -3.2854669, 2.3121364, -7.7434502, 7.1018934
1: -3.7775393, 4.0917711, -2.2581439, 2.5540643, -6.3316035, 6.3499150
2: -5.3876495, 4.0095453, -3.2469697, 2.5976362, -7.9852858, 7.2565150
3: -6.4850764, 3.2273605, -3.9351804, 2.0311718, -8.5162487, 7.1625409
4: -6.7883706, 4.1365519, -4.1189842, 2.7247391, -9.5131092, 8.2555361
5: -5.6790285, 3.4811852, -3.4368219, 2.2112696, -7.8902979, 6.9180069
6: -6.6366816, 3.8074152, -3.9622426, 2.5485334, -9.1852150, 7.7696581
7: -4.8902617, 4.8861866, -3.0648944, 3.0667505, -7.9570122, 7.9510813
8: -7.2463789, 3.6993084, -4.3799515, 2.3544874, -9.6008663, 8.0792599
9: -4.7286091, 4.7692227, -2.9633648, 2.9911404, -7.7197495, 7.7325878

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066073
time: 3.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066072
time: 3.04 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.0674763, 2.8777096, -5.9044561, 4.1473718, -8.2148476, 8.7821655
1: -2.7773356, 3.1218820, -4.1248484, 4.4328451, -7.2101808, 7.2467303
2: -4.0264611, 3.1180172, -5.8522348, 4.3242307, -8.3506918, 8.9702520
3: -4.8471227, 2.4828634, -7.0477638, 3.4853535, -8.3324757, 9.5306273
4: -5.0759230, 3.2510345, -7.3831129, 4.4480991, -9.5240221, 10.6341476
5: -4.2390509, 2.6797059, -6.1838140, 3.7598197, -7.9988708, 8.8635197
6: -4.9378076, 3.0720673, -7.2354507, 4.0683651, -9.0061722, 10.3075180
7: -3.7234621, 3.7077055, -5.2927752, 5.2955866, -9.0190487, 9.0004807
8: -5.4250007, 2.9023461, -7.8831029, 3.9905207, -9.4155216, 10.7854490
9: -3.6089909, 3.6207800, -5.1222954, 5.1565070, -8.7654982, 8.7430754

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.23 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
time: 2.06 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.4313140, 3.8164265, -6.0087199, 4.2197647, -9.6510792, 9.8251467
1: -3.7775393, 4.0917711, -4.2012739, 4.5072556, -8.2847948, 8.2930450
2: -5.3876495, 4.0095453, -5.9559159, 4.3928771, -9.7805271, 9.9654617
3: -6.4850764, 3.2273605, -7.1728597, 3.5424027, -10.0274792, 10.4002199
4: -6.7883706, 4.1365519, -7.5133233, 4.5158815, -11.3042526, 11.6498756
5: -5.6790285, 3.4811852, -6.2941003, 3.8213263, -9.5003548, 9.7752857
6: -6.6366816, 3.8074152, -7.3658819, 4.1267543, -10.7634354, 11.1732969
7: -4.8902617, 4.8861866, -5.3816962, 5.3854218, -10.2756834, 10.2678833
8: -7.2463789, 3.6993084, -8.0228271, 4.0542316, -11.3006105, 11.7221355
9: -4.7286091, 4.7692227, -5.2081914, 5.2435989, -9.9722080, 9.9774141

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055987, upper bound: 7.1055987
time: 2.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1055987, upper bound: 7.1055988
time: 2.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1073789, upper bound: 7.1073789
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1066072, upper bound: 7.1058193
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066073
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066072
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066073
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1058193, upper bound: 7.1066072
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1055988, upper bound: 7.1055987
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1055987, upper bound: 7.1055987
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.60
Output dim: 6, lower bound: -7.1055987, upper bound: 7.1055988

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.4147537, 1.0888404, -1.4147537, 1.0888404, -2.5035939, 2.5035939
1: -1.1065094, 1.1250544, -1.1065094, 1.1250544, -2.2315638, 2.2315638
2: -1.2957649, 1.3731751, -1.2957649, 1.3731751, -2.6689401, 2.6689401
3: -1.5482196, 0.9737427, -1.5482196, 0.9737427, -2.5219622, 2.5219622
4: -1.6589015, 1.4458973, -1.6589015, 1.4458973, -3.1047988, 3.1047988
5: -1.4666899, 1.1632832, -1.4666899, 1.1632832, -2.6299732, 2.6299732
6: -1.4416943, 1.5811431, -1.4416943, 1.5811431, -3.0228374, 3.0228374
7: -1.3833644, 1.4175160, -1.3833644, 1.4175160, -2.8008804, 2.8008804
8: -1.7159976, 1.2555535, -1.7159976, 1.2555535, -2.9715509, 2.9715509
9: -1.3292929, 1.3916898, -1.3292929, 1.3916898, -2.7209826, 2.7209826

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
time: 3.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998597
time: 4.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.4147537, 1.0888404, -2.7531009, 1.9244990, -3.3392527, 3.8419414
1: -1.1065094, 1.1250544, -1.9279807, 2.1453786, -3.2518880, 3.0530350
2: -1.2957649, 1.3731751, -2.6867716, 2.2426157, -3.5383806, 4.0599470
3: -1.5482196, 0.9737427, -3.2610307, 1.7218046, -3.2700243, 4.2347736
4: -1.6589015, 1.4458973, -3.4038484, 2.3624566, -4.0213580, 4.8497458
5: -1.4666899, 1.1632832, -2.8362594, 1.9072357, -3.3739257, 3.9995427
6: -1.4416943, 1.5811431, -3.2555003, 2.2518134, -3.6935077, 4.8366432
7: -1.3833644, 1.4175160, -2.5857768, 2.5900977, -3.9734621, 4.0032930
8: -1.7159976, 1.2555535, -3.6204588, 2.0194774, -3.7354751, 4.8760123
9: -1.3292929, 1.3916898, -2.4945264, 2.5285211, -3.8578138, 3.8862162

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1023069
time: 3.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0999247
time: 2.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -2.7531009, 1.9244990, -1.4147537, 1.0888404, -3.8419414, 3.3392527
1: -1.9279807, 2.1453786, -1.1065094, 1.1250544, -3.0530350, 3.2518880
2: -2.6867716, 2.2426157, -1.2957649, 1.3731751, -4.0599470, 3.5383806
3: -3.2610307, 1.7218046, -1.5482196, 0.9737427, -4.2347736, 3.2700243
4: -3.4038484, 2.3624566, -1.6589015, 1.4458973, -4.8497458, 4.0213580
5: -2.8362594, 1.9072357, -1.4666899, 1.1632832, -3.9995427, 3.3739257
6: -3.2555003, 2.2518134, -1.4416943, 1.5811431, -4.8366432, 3.6935077
7: -2.5857768, 2.5900977, -1.3833644, 1.4175160, -4.0032930, 3.9734621
8: -3.6204588, 2.0194774, -1.7159976, 1.2555535, -4.8760123, 3.7354751
9: -2.4945264, 2.5285211, -1.3292929, 1.3916898, -3.8862162, 3.8578138

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -2.7531009, 1.9244990, -2.7531009, 1.9244990, -4.6775999, 4.6775999
1: -1.9279807, 2.1453786, -1.9279807, 2.1453786, -4.0733595, 4.0733595
2: -2.6867716, 2.2426157, -2.6867716, 2.2426157, -4.9293871, 4.9293871
3: -3.2610307, 1.7218046, -3.2610307, 1.7218046, -4.9828353, 4.9828353
4: -3.4038484, 2.3624566, -3.4038484, 2.3624566, -5.7663050, 5.7663050
5: -2.8362594, 1.9072357, -2.8362594, 1.9072357, -4.7434950, 4.7434950
6: -3.2555003, 2.2518134, -3.2555003, 2.2518134, -5.5073137, 5.5073137
7: -2.5857768, 2.5900977, -2.5857768, 2.5900977, -5.1758747, 5.1758747
8: -3.6204588, 2.0194774, -3.6204588, 2.0194774, -5.6399364, 5.6399364
9: -2.4945264, 2.5285211, -2.4945264, 2.5285211, -5.0230474, 5.0230474

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020993
time: 3.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.1000677
time: 2.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.4147537, 1.0888404, -4.0674763, 2.8777096, -4.2924633, 5.1563168
1: -1.1065094, 1.1250544, -2.7773356, 3.1218820, -4.2283916, 3.9023900
2: -1.2957649, 1.3731751, -4.0264611, 3.1180172, -4.4137821, 5.3996363
3: -1.5482196, 0.9737427, -4.8471227, 2.4828634, -4.0310831, 5.8208656
4: -1.6589015, 1.4458973, -5.0759230, 3.2510345, -4.9099360, 6.5218201
5: -1.4666899, 1.1632832, -4.2390509, 2.6797059, -4.1463957, 5.4023342
6: -1.4416943, 1.5811431, -4.9378076, 3.0720673, -4.5137615, 6.5189505
7: -1.3833644, 1.4175160, -3.7234621, 3.7077055, -5.0910702, 5.1409779
8: -1.7159976, 1.2555535, -5.4250007, 2.9023461, -4.6183438, 6.6805544
9: -1.3292929, 1.3916898, -3.6089909, 3.6207800, -4.9500728, 5.0006809

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971582
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.4147537, 1.0888404, -5.4313140, 3.8164265, -5.2311802, 6.5201545
1: -1.1065094, 1.1250544, -3.7775393, 4.0917711, -5.1982803, 4.9025936
2: -1.2957649, 1.3731751, -5.3876495, 4.0095453, -5.3053102, 6.7608247
3: -1.5482196, 0.9737427, -6.4850764, 3.2273605, -4.7755799, 7.4588194
4: -1.6589015, 1.4458973, -6.7883706, 4.1365519, -5.7954531, 8.2342682
5: -1.4666899, 1.1632832, -5.6790285, 3.4811852, -4.9478750, 6.8423119
6: -1.4416943, 1.5811431, -6.6366816, 3.8074152, -5.2491093, 8.2178249
7: -1.3833644, 1.4175160, -4.8902617, 4.8861866, -6.2695513, 6.3077774
8: -1.7159976, 1.2555535, -7.2463789, 3.6993084, -5.4153061, 8.5019321
9: -1.3292929, 1.3916898, -4.7286091, 4.7692227, -6.0985155, 6.1202989

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0973448
time: 3.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949817
time: 3.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -2.7531009, 1.9244990, -4.0674763, 2.8777096, -5.6308107, 5.9919753
1: -1.9279807, 2.1453786, -2.7773356, 3.1218820, -5.0498629, 4.9227142
2: -2.6867716, 2.2426157, -4.0264611, 3.1180172, -5.8047886, 6.2690768
3: -3.2610307, 1.7218046, -4.8471227, 2.4828634, -5.7438941, 6.5689273
4: -3.4038484, 2.3624566, -5.0759230, 3.2510345, -6.6548829, 7.4383793
5: -2.8362594, 1.9072357, -4.2390509, 2.6797059, -5.5159655, 6.1462865
6: -3.2555003, 2.2518134, -4.9378076, 3.0720673, -6.3275676, 7.1896210
7: -2.5857768, 2.5900977, -3.7234621, 3.7077055, -6.2934823, 6.3135595
8: -3.6204588, 2.0194774, -5.4250007, 2.9023461, -6.5228052, 7.4444780
9: -2.4945264, 2.5285211, -3.6089909, 3.6207800, -6.1153064, 6.1375122

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971582
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
time: 3.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2.7531009, 1.9244990, -5.4313140, 3.8164265, -6.5695276, 7.3558130
1: -1.9279807, 2.1453786, -3.7775393, 4.0917711, -6.0197515, 5.9229178
2: -2.6867716, 2.2426157, -5.3876495, 4.0095453, -6.6963167, 7.6302652
3: -3.2610307, 1.7218046, -6.4850764, 3.2273605, -6.4883909, 8.2068806
4: -3.4038484, 2.3624566, -6.7883706, 4.1365519, -7.5404005, 9.1508274
5: -2.8362594, 1.9072357, -5.6790285, 3.4811852, -6.3174448, 7.5862641
6: -3.2555003, 2.2518134, -6.6366816, 3.8074152, -7.0629158, 8.8884945
7: -2.5857768, 2.5900977, -4.8902617, 4.8861866, -7.4719634, 7.4803591
8: -3.6204588, 2.0194774, -7.2463789, 3.6993084, -7.3197670, 9.2658558
9: -2.4945264, 2.5285211, -4.7286091, 4.7692227, -7.2637491, 7.2571301

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0972140
time: 3.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0951309
time: 2.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0674763, 2.8777096, -1.4147537, 1.0888404, -5.1563168, 4.2924633
1: -2.7773356, 3.1218820, -1.1065094, 1.1250544, -3.9023900, 4.2283916
2: -4.0264611, 3.1180172, -1.2957649, 1.3731751, -5.3996363, 4.4137821
3: -4.8471227, 2.4828634, -1.5482196, 0.9737427, -5.8208656, 4.0310831
4: -5.0759230, 3.2510345, -1.6589015, 1.4458973, -6.5218201, 4.9099360
5: -4.2390509, 2.6797059, -1.4666899, 1.1632832, -5.4023342, 4.1463957
6: -4.9378076, 3.0720673, -1.4416943, 1.5811431, -6.5189505, 4.5137615
7: -3.7234621, 3.7077055, -1.3833644, 1.4175160, -5.1409779, 5.0910702
8: -5.4250007, 2.9023461, -1.7159976, 1.2555535, -6.6805544, 4.6183438
9: -3.6089909, 3.6207800, -1.3292929, 1.3916898, -5.0006809, 4.9500728

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1007620
time: 2.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
time: 2.31 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.0674763, 2.8777096, -2.7531009, 1.9244990, -5.9919753, 5.6308107
1: -2.7773356, 3.1218820, -1.9279807, 2.1453786, -4.9227142, 5.0498629
2: -4.0264611, 3.1180172, -2.6867716, 2.2426157, -6.2690768, 5.8047886
3: -4.8471227, 2.4828634, -3.2610307, 1.7218046, -6.5689273, 5.7438941
4: -5.0759230, 3.2510345, -3.4038484, 2.3624566, -7.4383793, 6.6548829
5: -4.2390509, 2.6797059, -2.8362594, 1.9072357, -6.1462865, 5.5159655
6: -4.9378076, 3.0720673, -3.2555003, 2.2518134, -7.1896210, 6.3275676
7: -3.7234621, 3.7077055, -2.5857768, 2.5900977, -6.3135595, 6.2934823
8: -5.4250007, 2.9023461, -3.6204588, 2.0194774, -7.4444780, 6.5228052
9: -3.6089909, 3.6207800, -2.4945264, 2.5285211, -6.1375122, 6.1153064

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1010712
time: 3.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0976299
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.4313140, 3.8164265, -1.4147537, 1.0888404, -6.5201545, 5.2311802
1: -3.7775393, 4.0917711, -1.1065094, 1.1250544, -4.9025936, 5.1982803
2: -5.3876495, 4.0095453, -1.2957649, 1.3731751, -6.7608247, 5.3053102
3: -6.4850764, 3.2273605, -1.5482196, 0.9737427, -7.4588194, 4.7755799
4: -6.7883706, 4.1365519, -1.6589015, 1.4458973, -8.2342682, 5.7954531
5: -5.6790285, 3.4811852, -1.4666899, 1.1632832, -6.8423119, 4.9478750
6: -6.6366816, 3.8074152, -1.4416943, 1.5811431, -8.2178249, 5.2491093
7: -4.8902617, 4.8861866, -1.3833644, 1.4175160, -6.3077774, 6.2695513
8: -7.2463789, 3.6993084, -1.7159976, 1.2555535, -8.5019321, 5.4153061
9: -4.7286091, 4.7692227, -1.3292929, 1.3916898, -6.1202989, 6.0985155

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1007620
time: 2.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.4313140, 3.8164265, -2.7531009, 1.9244990, -7.3558130, 6.5695276
1: -3.7775393, 4.0917711, -1.9279807, 2.1453786, -5.9229178, 6.0197515
2: -5.3876495, 4.0095453, -2.6867716, 2.2426157, -7.6302652, 6.6963167
3: -6.4850764, 3.2273605, -3.2610307, 1.7218046, -8.2068806, 6.4883909
4: -6.7883706, 4.1365519, -3.4038484, 2.3624566, -9.1508274, 7.5404005
5: -5.6790285, 3.4811852, -2.8362594, 1.9072357, -7.5862641, 6.3174448
6: -6.6366816, 3.8074152, -3.2555003, 2.2518134, -8.8884945, 7.0629158
7: -4.8902617, 4.8861866, -2.5857768, 2.5900977, -7.4803591, 7.4719634
8: -7.2463789, 3.6993084, -3.6204588, 2.0194774, -9.2658558, 7.3197670
9: -4.7286091, 4.7692227, -2.4945264, 2.5285211, -7.2571301, 7.2637491

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1007907
time: 2.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0977791
time: 2.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0674763, 2.8777096, -4.0674582, 2.8774748, -6.9449511, 6.9451675
1: -2.7773356, 3.1218820, -2.7771821, 3.1218820, -5.8992176, 5.8990641
2: -4.0264611, 3.1180172, -4.0264230, 3.1179724, -7.1444335, 7.1444402
3: -4.8471227, 2.4828634, -4.8434372, 2.4828491, -7.3299718, 7.3263006
4: -5.0759230, 3.2510345, -5.0759029, 3.2510345, -8.3269577, 8.3269377
5: -4.2390509, 2.6797059, -4.2390509, 2.6791158, -6.9181666, 6.9187565
6: -4.9378076, 3.0720673, -4.9378076, 3.0718241, -8.0096321, 8.0098743
7: -3.7234621, 3.7077055, -3.7234440, 3.7070456, -7.4305077, 7.4311495
8: -5.4250007, 2.9023461, -5.4249887, 2.9019449, -8.3269453, 8.3273354
9: -3.6089909, 3.6207800, -3.6089909, 3.6197283, -7.2287192, 7.2297707

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0969279
time: 2.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
time: 2.32 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.0674763, 2.8777096, -5.4263535, 3.8158021, -7.8832784, 8.3040628
1: -2.7773356, 3.1218820, -3.7753417, 4.0917711, -6.8691068, 6.8972235
2: -4.0264611, 3.1180172, -5.3774166, 4.0094271, -8.0358887, 8.4954338
3: -4.8471227, 2.4828634, -6.4753261, 3.2234309, -8.0705538, 8.9581890
4: -5.0759230, 3.2510345, -6.7870622, 4.1365519, -9.2124748, 10.0380964
5: -4.2390509, 2.6797059, -5.6790285, 3.4779925, -7.7170434, 8.3587341
6: -4.9378076, 3.0720673, -6.6366816, 3.8040371, -8.7418442, 9.7087488
7: -3.7234621, 3.7077055, -4.8853655, 4.8844371, -8.6078987, 8.5930710
8: -5.4250007, 2.9023461, -7.2433844, 3.6982622, -9.1232624, 10.1457310
9: -3.6089909, 3.6207800, -4.7286091, 4.7575555, -8.3665466, 8.3493891

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0971312
time: 2.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0942078
time: 2.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.4313140, 3.8164265, -4.0674582, 2.8774748, -8.3087883, 7.8838844
1: -3.7775393, 4.0917711, -2.7771821, 3.1218820, -6.8994212, 6.8689532
2: -5.3876495, 4.0095453, -4.0264230, 3.1179724, -8.5056219, 8.0359688
3: -6.4850764, 3.2273605, -4.8434372, 2.4828491, -8.9679260, 8.0707979
4: -6.7883706, 4.1365519, -5.0759029, 3.2510345, -10.0394049, 9.2124548
5: -5.6790285, 3.4811852, -4.2390509, 2.6791158, -8.3581448, 7.7202358
6: -6.6366816, 3.8074152, -4.9378076, 3.0718241, -9.7085056, 8.7452230
7: -4.8902617, 4.8861866, -3.7234440, 3.7070456, -8.5973072, 8.6096306
8: -7.2463789, 3.6993084, -5.4249887, 2.9019449, -10.1483240, 9.1242971
9: -4.7286091, 4.7692227, -3.6089909, 3.6197283, -8.3483372, 8.3782139

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0969279
time: 2.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
time: 3.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.4313140, 3.8164265, -5.4263535, 3.8158021, -9.2471161, 9.2427797
1: -3.7775393, 4.0917711, -3.7753417, 4.0917711, -7.8693104, 7.8671131
2: -5.3876495, 4.0095453, -5.3774166, 4.0094271, -9.3970766, 9.3869619
3: -6.4850764, 3.2273605, -6.4753261, 3.2234309, -9.7085075, 9.7026863
4: -6.7883706, 4.1365519, -6.7870622, 4.1365519, -10.9249229, 10.9236145
5: -5.6790285, 3.4811852, -5.6790285, 3.4779925, -9.1570206, 9.1602135
6: -6.6366816, 3.8074152, -6.6366816, 3.8040371, -10.4407187, 10.4440966
7: -4.8902617, 4.8861866, -4.8853655, 4.8844371, -9.7746983, 9.7715521
8: -7.2463789, 3.6993084, -7.2433844, 3.6982622, -10.9446411, 10.9426928
9: -4.7286091, 4.7692227, -4.7286091, 4.7575555, -9.4861641, 9.4978313

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0969715
time: 2.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0943763
time: 2.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.28 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998597
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1023069
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0999247
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020993
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0999247, upper bound: 7.1000677
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971582
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0973448
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949817
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971582
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0972140
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0951309
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1007620
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1010712
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0976299
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1007620
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.1000878, upper bound: 7.1007907
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0977791
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0969279
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0971312
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0942078
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0969279
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0990074, upper bound: 7.0969715
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.28
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0943763

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -1.4147537, 1.0888404, -1.9131303, 2.1907248
1: -0.7593619, 0.7365588, -1.1065094, 1.1250544, -1.8844163, 1.8430682
2: -0.7684594, 0.9860382, -1.2957649, 1.3731751, -2.1416345, 2.2818031
3: -0.7953699, 0.6663919, -1.5482196, 0.9737427, -1.7691126, 2.2146115
4: -0.9501143, 1.0594401, -1.6589015, 1.4458973, -2.3960116, 2.7183416
5: -0.9043361, 0.8626598, -1.4666899, 1.1632832, -2.0676193, 2.3293498
6: -0.6470271, 1.4070647, -1.4416943, 1.5811431, -2.2281704, 2.8487589
7: -0.9023290, 0.9419560, -1.3833644, 1.4175160, -2.3198450, 2.3253205
8: -0.9998023, 0.9381140, -1.7159976, 1.2555535, -2.2553558, 2.6541116
9: -0.8435787, 0.9448409, -1.3292929, 1.3916898, -2.2352686, 2.2741337

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998598
time: 3.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998598
time: 3.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -1.3215413, 1.0385540, -4.3148918, 3.4040260
1: -2.1758492, 2.4862385, -1.0534830, 1.0558715, -3.2317207, 3.5397215
2: -3.0691266, 2.5236025, -1.2045491, 1.3149843, -4.3841109, 3.7281516
3: -3.9400258, 1.9516809, -1.4273330, 0.9238526, -4.8638783, 3.3790140
4: -3.8988452, 2.6804190, -1.5456346, 1.3843467, -5.2831917, 4.2260537
5: -3.2787127, 2.1279092, -1.3750339, 1.1153960, -4.3941088, 3.5029430
6: -3.9502568, 2.3441942, -1.3137641, 1.5461588, -5.4964156, 3.6579583
7: -3.0190184, 2.9432998, -1.3021886, 1.3400869, -4.3591051, 4.2454882
8: -4.0411730, 2.2886350, -1.5995424, 1.2025629, -5.2437358, 3.8881774
9: -2.8845158, 2.8212209, -1.2505456, 1.3205893, -4.2051048, 4.0717664

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998470, upper bound: 7.0998597
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998597
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -2.7531009, 1.9244990, -2.7487891, 3.5290723
1: -0.7593619, 0.7365588, -1.9279807, 2.1453786, -2.9047406, 2.6645393
2: -0.7684594, 0.9860382, -2.6867716, 2.2426157, -3.0110750, 3.6728098
3: -0.7953699, 0.6663919, -3.2610307, 1.7218046, -2.5171745, 3.9274225
4: -0.9501143, 1.0594401, -3.4038484, 2.3624566, -3.3125708, 4.4632883
5: -0.9043361, 0.8626598, -2.8362594, 1.9072357, -2.8115718, 3.6989193
6: -0.6470271, 1.4070647, -3.2555003, 2.2518134, -2.8988404, 4.6625652
7: -0.9023290, 0.9419560, -2.5857768, 2.5900977, -3.4924266, 3.5277328
8: -0.9998023, 0.9381140, -3.6204588, 2.0194774, -3.0192797, 4.5585728
9: -0.8435787, 0.9448409, -2.4945264, 2.5285211, -3.3720999, 3.4393673

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
time: 2.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
time: 2.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -2.6521170, 1.8529329, -5.1292706, 4.7346020
1: -2.1758492, 2.4862385, -1.8652141, 2.0676956, -4.2435446, 4.3514528
2: -3.0691266, 2.5236025, -2.5795922, 2.1758349, -5.2449617, 5.1031947
3: -3.9400258, 1.9516809, -3.1323738, 1.6631525, -5.6031780, 5.0840549
4: -3.8988452, 2.6804190, -3.2678645, 2.2940652, -6.1929102, 5.9482832
5: -3.2787127, 2.1279092, -2.7281613, 1.8490180, -5.1277308, 4.8560705
6: -3.9502568, 2.3441942, -3.1232674, 2.1946571, -6.1449137, 5.4674616
7: -3.0190184, 2.9432998, -2.4941282, 2.5011022, -5.5201206, 5.4374280
8: -4.0411730, 2.2886350, -3.4760296, 1.9565072, -5.9976802, 5.7646646
9: -2.8845158, 2.8212209, -2.4055524, 2.4410515, -5.3255672, 5.2267733

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
time: 2.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
time: 2.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -1.4147537, 1.0888404, -3.1359463, 2.8626695
1: -1.4879133, 1.6022439, -1.1065094, 1.1250544, -2.6129675, 2.7087533
2: -1.9315817, 1.7790974, -1.2957649, 1.3731751, -3.3047569, 3.0748625
3: -2.3510928, 1.3185607, -1.5482196, 0.9737427, -3.3248355, 2.8667803
4: -2.4359283, 1.8813576, -1.6589015, 1.4458973, -3.8818257, 3.5402589
5: -2.1040509, 1.5007353, -1.4666899, 1.1632832, -3.2673340, 2.9674253
6: -2.3195608, 1.8668721, -1.4416943, 1.5811431, -3.9007039, 3.3085663
7: -1.9417379, 1.9663347, -1.3833644, 1.4175160, -3.3592539, 3.3496990
8: -2.5997360, 1.6031470, -1.7159976, 1.2555535, -3.8552895, 3.3191447
9: -1.8700268, 1.9104283, -1.3292929, 1.3916898, -3.2617166, 3.2397213

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.2181687, 3.6797018, -1.3215413, 1.0385540, -6.2567225, 5.0012431
1: -3.4342718, 4.0089335, -1.0534830, 1.0558715, -4.4901433, 5.0624166
2: -5.2342386, 3.8684192, -1.2045491, 1.3149843, -6.5492229, 5.0729685
3: -6.3400049, 3.1585281, -1.4273330, 0.9238526, -7.2638574, 4.5858612
4: -6.6851840, 4.0211673, -1.5456346, 1.3843467, -8.0695305, 5.5668020
5: -5.5670929, 3.3006294, -1.3750339, 1.1153960, -6.6824889, 4.6756630
6: -6.4477587, 3.6439188, -1.3137641, 1.5461588, -7.9939175, 4.9576826
7: -4.7742691, 4.7608337, -1.3021886, 1.3400869, -6.1143560, 6.0630226
8: -7.0840087, 3.5179584, -1.5995424, 1.2025629, -8.2865715, 5.1175008
9: -4.6274099, 4.6256738, -1.2505456, 1.3205893, -5.9479990, 5.8762193

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998959, upper bound: 7.0999381
time: 2.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
time: 2.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -2.7531009, 1.9244990, -3.9716048, 4.2010164
1: -1.4879133, 1.6022439, -1.9279807, 2.1453786, -3.6332917, 3.5302246
2: -1.9315817, 1.7790974, -2.6867716, 2.2426157, -4.1741972, 4.4658689
3: -2.3510928, 1.3185607, -3.2610307, 1.7218046, -4.0728974, 4.5795913
4: -2.4359283, 1.8813576, -3.4038484, 2.3624566, -4.7983847, 5.2852058
5: -2.1040509, 1.5007353, -2.8362594, 1.9072357, -4.0112867, 4.3369946
6: -2.3195608, 1.8668721, -3.2555003, 2.2518134, -4.5713739, 5.1223726
7: -1.9417379, 1.9663347, -2.5857768, 2.5900977, -4.5318356, 4.5521116
8: -2.5997360, 1.6031470, -3.6204588, 2.0194774, -4.6192131, 5.2236061
9: -1.8700268, 1.9104283, -2.4945264, 2.5285211, -4.3985481, 4.4049549

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
time: 2.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
time: 2.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2181687, 3.6797018, -2.6521170, 1.8529329, -7.0711017, 6.3318186
1: -3.4342718, 4.0089335, -1.8652141, 2.0676956, -5.5019674, 5.8741474
2: -5.2342386, 3.8684192, -2.5795922, 2.1758349, -7.4100733, 6.4480114
3: -6.3400049, 3.1585281, -3.1323738, 1.6631525, -8.0031576, 6.2909021
4: -6.6851840, 4.0211673, -3.2678645, 2.2940652, -8.9792490, 7.2890320
5: -5.5670929, 3.3006294, -2.7281613, 1.8490180, -7.4161110, 6.0287905
6: -6.4477587, 3.6439188, -3.1232674, 2.1946571, -8.6424160, 6.7671862
7: -4.7742691, 4.7608337, -2.4941282, 2.5011022, -7.2753716, 7.2549620
8: -7.0840087, 3.5179584, -3.4760296, 1.9565072, -9.0405159, 6.9939880
9: -4.6274099, 4.6256738, -2.4055524, 2.4410515, -7.0684614, 7.0312262

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000534, upper bound: 7.1000677
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
time: 2.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -4.0674763, 2.8777096, -3.7019997, 4.8434477
1: -0.7593619, 0.7365588, -2.7773356, 3.1218820, -3.8812439, 3.5138946
2: -0.7684594, 0.9860382, -4.0264611, 3.1180172, -3.8864765, 5.0124993
3: -0.7953699, 0.6663919, -4.8471227, 2.4828634, -3.2782333, 5.5135145
4: -0.9501143, 1.0594401, -5.0759230, 3.2510345, -4.2011490, 6.1353631
5: -0.9043361, 0.8626598, -4.2390509, 2.6797059, -3.5840421, 5.1017108
6: -0.6470271, 1.4070647, -4.9378076, 3.0720673, -3.7190943, 6.3448725
7: -0.9023290, 0.9419560, -3.7234621, 3.7077055, -4.6100345, 4.6654181
8: -0.9998023, 0.9381140, -5.4250007, 2.9023461, -3.9021485, 6.3631148
9: -0.8435787, 0.9448409, -3.6089909, 3.6207800, -4.4643588, 4.5538321

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
time: 2.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
time: 3.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -3.9496896, 2.7958455, -6.0721831, 6.0321741
1: -2.1758492, 2.4862385, -2.6909211, 3.0387743, -5.2146235, 5.1771593
2: -3.0691266, 2.5236025, -3.9092083, 3.0409184, -6.1100450, 6.4328108
3: -3.9400258, 1.9516809, -4.7046313, 2.4178717, -6.3578978, 6.6563120
4: -3.8988452, 2.6804190, -4.9304795, 3.1744716, -7.0733166, 7.6108985
5: -3.2787127, 2.1279092, -4.1160855, 2.6103840, -5.8890967, 6.2439947
6: -3.9502568, 2.3441942, -4.7932444, 3.0072055, -6.9574623, 7.1374388
7: -3.0190184, 2.9432998, -3.6228197, 3.6067400, -6.6257582, 6.5661192
8: -4.0411730, 2.2886350, -5.2694817, 2.8301940, -6.8713670, 7.5581169
9: -2.8845158, 2.8212209, -3.5118060, 3.5231705, -6.4076862, 6.3330269

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0973767, upper bound: 7.0949013
time: 3.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
time: 2.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -5.4313140, 3.8164265, -4.6407166, 6.2072854
1: -0.7593619, 0.7365588, -3.7775393, 4.0917711, -4.8511329, 4.5140982
2: -0.7684594, 0.9860382, -5.3876495, 4.0095453, -4.7780046, 6.3736877
3: -0.7953699, 0.6663919, -6.4850764, 3.2273605, -4.0227304, 7.1514683
4: -0.9501143, 1.0594401, -6.7883706, 4.1365519, -5.0866661, 7.8478107
5: -0.9043361, 0.8626598, -5.6790285, 3.4811852, -4.3855214, 6.5416884
6: -0.6470271, 1.4070647, -6.6366816, 3.8074152, -4.4544425, 8.0437460
7: -0.9023290, 0.9419560, -4.8902617, 4.8861866, -5.7885156, 5.8322177
8: -0.9998023, 0.9381140, -7.2463789, 3.6993084, -4.6991105, 8.1844931
9: -0.8435787, 0.9448409, -4.7286091, 4.7692227, -5.6128016, 5.6734500

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976572, upper bound: 7.0949817
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976572, upper bound: 7.0949817
time: 3.10 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -5.3165779, 3.7370844, -7.0134220, 7.3990626
1: -2.1758492, 2.4862385, -3.6935704, 4.0112262, -6.1870756, 6.1798086
2: -3.0691266, 2.5236025, -5.2732911, 3.9347608, -7.0038872, 7.7968936
3: -3.9400258, 1.9516809, -6.3467689, 3.1640866, -7.1041126, 8.2984495
4: -3.8988452, 2.6804190, -6.6470652, 4.0623679, -7.9612131, 9.3274841
5: -3.2787127, 2.1279092, -5.5598159, 3.4137661, -6.6924791, 7.6877251
6: -3.9502568, 2.3441942, -6.4959188, 3.7444239, -7.6946807, 8.8401127
7: -3.0190184, 2.9432998, -4.7923841, 4.7882700, -7.8072882, 7.7356839
8: -4.0411730, 2.2886350, -7.0954123, 3.6292741, -7.6704473, 9.3840475
9: -2.8845158, 2.8212209, -4.6344328, 4.6738749, -7.5583906, 7.4556537

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975070, upper bound: 7.0949817
time: 3.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976572, upper bound: 7.0949817
time: 2.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -4.0674763, 2.8777096, -4.9248152, 5.5153923
1: -1.4879133, 1.6022439, -2.7773356, 3.1218820, -4.6097951, 4.3795795
2: -1.9315817, 1.7790974, -4.0264611, 3.1180172, -5.0495987, 5.8055587
3: -2.3510928, 1.3185607, -4.8471227, 2.4828634, -4.8339562, 6.1656833
4: -2.4359283, 1.8813576, -5.0759230, 3.2510345, -5.6869631, 6.9572806
5: -2.1040509, 1.5007353, -4.2390509, 2.6797059, -4.7837567, 5.7397861
6: -2.3195608, 1.8668721, -4.9378076, 3.0720673, -5.3916283, 6.8046799
7: -1.9417379, 1.9663347, -3.7234621, 3.7077055, -5.6494436, 5.6897969
8: -2.5997360, 1.6031470, -5.4250007, 2.9023461, -5.5020819, 7.0281477
9: -1.8700268, 1.9104283, -3.6089909, 3.6207800, -5.4908066, 5.5194192

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
time: 2.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
time: 2.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.2181687, 3.6797018, -3.9496896, 2.7958455, -8.0140142, 7.6293917
1: -3.4342718, 4.0089335, -2.6909211, 3.0387743, -6.4730463, 6.6998549
2: -5.2342386, 3.8684192, -3.9092083, 3.0409184, -8.2751570, 7.7776275
3: -6.3400049, 3.1585281, -4.7046313, 2.4178717, -8.7578764, 7.8631592
4: -6.6851840, 4.0211673, -4.9304795, 3.1744716, -9.8596554, 8.9516468
5: -5.5670929, 3.3006294, -4.1160855, 2.6103840, -8.1774769, 7.4167147
6: -6.4477587, 3.6439188, -4.7932444, 3.0072055, -9.4549637, 8.4371634
7: -4.7742691, 4.7608337, -3.6228197, 3.6067400, -8.3810091, 8.3836536
8: -7.0840087, 3.5179584, -5.2694817, 2.8301940, -9.9142027, 8.7874403
9: -4.6274099, 4.6256738, -3.5118060, 3.5231705, -8.1505804, 8.1374798

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0974159, upper bound: 7.0949797
time: 2.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
time: 2.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -5.4313140, 3.8164265, -5.8635321, 6.8792295
1: -1.4879133, 1.6022439, -3.7775393, 4.0917711, -5.5796843, 5.3797832
2: -1.9315817, 1.7790974, -5.3876495, 4.0095453, -5.9411268, 7.1667471
3: -2.3510928, 1.3185607, -6.4850764, 3.2273605, -5.5784531, 7.8036370
4: -2.4359283, 1.8813576, -6.7883706, 4.1365519, -6.5724802, 8.6697283
5: -2.1040509, 1.5007353, -5.6790285, 3.4811852, -5.5852361, 7.1797638
6: -2.3195608, 1.8668721, -6.6366816, 3.8074152, -6.1269760, 8.5035534
7: -1.9417379, 1.9663347, -4.8902617, 4.8861866, -6.8279247, 6.8565965
8: -2.5997360, 1.6031470, -7.2463789, 3.6993084, -6.2990446, 8.8495255
9: -1.8700268, 1.9104283, -4.7286091, 4.7692227, -6.6392498, 6.6390371

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0977776, upper bound: 7.0951309
time: 2.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0977776, upper bound: 7.0951309
time: 3.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2181687, 3.6797018, -5.3165779, 3.7370844, -8.9552536, 8.9962797
1: -3.4342718, 4.0089335, -3.6935704, 4.0112262, -7.4454980, 7.7025042
2: -5.2342386, 3.8684192, -5.2732911, 3.9347608, -9.1689997, 9.1417103
3: -6.3400049, 3.1585281, -6.3467689, 3.1640866, -9.5040913, 9.5052967
4: -6.6851840, 4.0211673, -6.6470652, 4.0623679, -10.7475519, 10.6682320
5: -5.5670929, 3.3006294, -5.5598159, 3.4137661, -8.9808588, 8.8604450
6: -6.4477587, 3.6439188, -6.4959188, 3.7444239, -10.1921825, 10.1398373
7: -4.7742691, 4.7608337, -4.7923841, 4.7882700, -9.5625391, 9.5532179
8: -7.0840087, 3.5179584, -7.0954123, 3.6292741, -10.7132826, 10.6133709
9: -4.6274099, 4.6256738, -4.6344328, 4.6738749, -9.3012848, 9.2601070

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0976372, upper bound: 7.0951309
time: 2.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0977776, upper bound: 7.0951309
time: 2.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.2629774, 2.3178856, -1.4147537, 1.0888404, -4.3518176, 3.7326393
1: -2.2444396, 2.5407939, -1.1065094, 1.1250544, -3.3694940, 3.6473033
2: -3.2150350, 2.5943975, -1.2957649, 1.3731751, -4.5882101, 3.8901625
3: -3.8694959, 2.0315437, -1.5482196, 0.9737427, -4.8432388, 3.5797634
4: -4.0633783, 2.7323642, -1.6589015, 1.4458973, -5.5092754, 4.3912659
5: -3.3813291, 2.2124634, -1.4666899, 1.1632832, -4.5446124, 3.6791534
6: -3.9359269, 2.6354151, -1.4416943, 1.5811431, -5.5170698, 4.0771093
7: -3.0285966, 3.0167100, -1.3833644, 1.4175160, -4.4461126, 4.4000745
8: -4.3410654, 2.4173012, -1.7159976, 1.2555535, -5.5966187, 4.1332989
9: -2.9352677, 2.9538395, -1.3292929, 1.3916898, -4.3269577, 4.2831326

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
time: 2.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1953211, 4.3345661, -1.3215413, 1.0385540, -7.2338753, 5.6561074
1: -4.3170023, 4.6164837, -1.0534830, 1.0558715, -5.3728738, 5.6699667
2: -6.1278896, 4.5010343, -1.2045491, 1.3149843, -7.4428740, 5.7055836
3: -7.3889561, 3.6548228, -1.4273330, 0.9238526, -8.3128090, 5.0821557
4: -7.7128749, 4.6105943, -1.5456346, 1.3843467, -9.0972214, 6.1562290
5: -6.4589248, 3.9231412, -1.3750339, 1.1153960, -7.5743208, 5.2981749
6: -7.5325246, 4.2447419, -1.3137641, 1.5461588, -9.0786839, 5.5585060
7: -5.5208154, 5.5095692, -1.3021886, 1.3400869, -6.8609023, 6.8117580
8: -8.2391109, 4.1719885, -1.5995424, 1.2025629, -9.4416742, 5.7715311
9: -5.3339462, 5.3843799, -1.2505456, 1.3205893, -6.6545353, 6.6349254

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0948753, upper bound: 7.0975675
time: 2.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
time: 2.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.2629774, 2.3178856, -2.7531009, 1.9244990, -5.1874762, 5.0709867
1: -2.2444396, 2.5407939, -1.9279807, 2.1453786, -4.3898182, 4.4687748
2: -3.2150350, 2.5943975, -2.6867716, 2.2426157, -5.4576507, 5.2811689
3: -3.8694959, 2.0315437, -3.2610307, 1.7218046, -5.5913005, 5.2925744
4: -4.0633783, 2.7323642, -3.4038484, 2.3624566, -6.4258347, 6.1362123
5: -3.3813291, 2.2124634, -2.8362594, 1.9072357, -5.2885647, 5.0487227
6: -3.9359269, 2.6354151, -3.2555003, 2.2518134, -6.1877403, 5.8909154
7: -3.0285966, 3.0167100, -2.5857768, 2.5900977, -5.6186943, 5.6024866
8: -4.3410654, 2.4173012, -3.6204588, 2.0194774, -6.3605428, 6.0377598
9: -2.9352677, 2.9538395, -2.4945264, 2.5285211, -5.4637890, 5.4483662

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949797, upper bound: 7.0976299
time: 2.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949797, upper bound: 7.0976299
time: 2.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1953211, 4.3345661, -2.6521170, 1.8529329, -8.0482540, 6.9866829
1: -4.3170023, 4.6164837, -1.8652141, 2.0676956, -6.3846979, 6.4816980
2: -6.1278896, 4.5010343, -2.5795922, 2.1758349, -8.3037243, 7.0806265
3: -7.3889561, 3.6548228, -3.1323738, 1.6631525, -9.0521088, 6.7871966
4: -7.7128749, 4.6105943, -3.2678645, 2.2940652, -10.0069399, 7.8784590
5: -6.4589248, 3.9231412, -2.7281613, 1.8490180, -8.3079424, 6.6513023
6: -7.5325246, 4.2447419, -3.1232674, 2.1946571, -9.7271814, 7.3680096
7: -5.5208154, 5.5095692, -2.4941282, 2.5011022, -8.0219173, 8.0036974
8: -8.2391109, 4.1719885, -3.4760296, 1.9565072, -10.1956177, 7.6480179
9: -5.3339462, 5.3843799, -2.4055524, 2.4410515, -7.7749977, 7.7899323

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949790, upper bound: 7.0976299
time: 2.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949797, upper bound: 7.0976299
time: 2.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.5982270, 3.2413566, -1.4147537, 1.0888404, -5.6870675, 4.6561103
1: -3.1697054, 3.5041323, -1.1065094, 1.1250544, -4.2947598, 4.6106415
2: -4.5571985, 3.4658110, -1.2957649, 1.3731751, -5.9303737, 4.7615757
3: -5.4831147, 2.7663198, -1.5482196, 0.9737427, -6.4568577, 4.3145394
4: -5.7625318, 3.5996904, -1.6589015, 1.4458973, -7.2084293, 5.2585917
5: -4.8105569, 2.9916859, -1.4666899, 1.1632832, -5.9738402, 4.4583759
6: -5.6117258, 3.3567023, -1.4416943, 1.5811431, -7.1928692, 4.7983966
7: -4.1799908, 4.1736808, -1.3833644, 1.4175160, -5.5975065, 5.5570450
8: -6.1474762, 3.1974916, -1.7159976, 1.2555535, -7.4030294, 4.9134893
9: -4.0446091, 4.0761676, -1.3292929, 1.3916898, -5.4362988, 5.4054604

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
time: 3.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
time: 3.07 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.9326782, 5.5283484, -1.3215413, 1.0385540, -8.9712324, 6.8498898
1: -5.5904951, 5.8529410, -1.0534830, 1.0558715, -6.6463666, 6.9064240
2: -7.8620582, 5.6379375, -1.2045491, 1.3149843, -9.1770420, 6.8424864
3: -9.4784203, 4.6039939, -1.4273330, 0.9238526, -10.4022732, 6.0313268
4: -9.9027557, 5.7387829, -1.5456346, 1.3843467, -11.2871027, 7.2844176
5: -8.2941227, 4.9457088, -1.3750339, 1.1153960, -9.4095192, 6.3207426
6: -9.7008581, 5.2041874, -1.3137641, 1.5461588, -11.2470169, 6.5179515
7: -7.0097895, 7.0129094, -1.3021886, 1.3400869, -8.3498764, 8.3150978
8: -10.5603228, 5.1728263, -1.5995424, 1.2025629, -11.7628860, 6.7723684
9: -6.7609205, 6.8487992, -1.2505456, 1.3205893, -8.0815096, 8.0993452

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949423, upper bound: 7.0976572
time: 2.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
time: 2.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.5982270, 3.2413566, -2.7531009, 1.9244990, -6.5227261, 5.9944572
1: -3.1697054, 3.5041323, -1.9279807, 2.1453786, -5.3150840, 5.4321127
2: -4.5571985, 3.4658110, -2.6867716, 2.2426157, -6.7998142, 6.1525826
3: -5.4831147, 2.7663198, -3.2610307, 1.7218046, -7.2049193, 6.0273504
4: -5.7625318, 3.5996904, -3.4038484, 2.3624566, -8.1249886, 7.0035391
5: -4.8105569, 2.9916859, -2.8362594, 1.9072357, -6.7177925, 5.8279452
6: -5.6117258, 3.3567023, -3.2555003, 2.2518134, -7.8635392, 6.6122026
7: -4.1799908, 4.1736808, -2.5857768, 2.5900977, -6.7700882, 6.7594576
8: -6.1474762, 3.1974916, -3.6204588, 2.0194774, -8.1669540, 6.8179502
9: -4.0446091, 4.0761676, -2.4945264, 2.5285211, -6.5731301, 6.5706940

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951156, upper bound: 7.0977791
time: 2.24 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951156, upper bound: 7.0977791
time: 2.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.9326782, 5.5283484, -2.6521170, 1.8529329, -9.7856112, 8.1804657
1: -5.5904951, 5.8529410, -1.8652141, 2.0676956, -7.6581907, 7.7181549
2: -7.8620582, 5.6379375, -2.5795922, 2.1758349, -10.0378933, 8.2175293
3: -9.4784203, 4.6039939, -3.1323738, 1.6631525, -11.1415730, 7.7363677
4: -9.9027557, 5.7387829, -3.2678645, 2.2940652, -12.1968212, 9.0066471
5: -8.2941227, 4.9457088, -2.7281613, 1.8490180, -10.1431408, 7.6738701
6: -9.7008581, 5.2041874, -3.1232674, 2.1946571, -11.8955154, 8.3274546
7: -7.0097895, 7.0129094, -2.4941282, 2.5011022, -9.5108919, 9.5070381
8: -10.5603228, 5.1728263, -3.4760296, 1.9565072, -12.5168304, 8.6488562
9: -6.7609205, 6.8487992, -2.4055524, 2.4410515, -9.2019720, 9.2543516

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951068, upper bound: 7.0977791
time: 1.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951156, upper bound: 7.0977791
time: 2.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.2629774, 2.3178856, -4.0674582, 2.8774748, -6.1404524, 6.3853436
1: -2.2444396, 2.5407939, -2.7771821, 3.1218820, -5.3663216, 5.3179760
2: -3.2150350, 2.5943975, -4.0264230, 3.1179724, -6.3330073, 6.6208205
3: -3.8694959, 2.0315437, -4.8434372, 2.4828491, -6.3523450, 6.8749809
4: -4.0633783, 2.7323642, -5.0759029, 3.2510345, -7.3144131, 7.8082671
5: -3.3813291, 2.2124634, -4.2390509, 2.6791158, -6.0604448, 6.4515142
6: -3.9359269, 2.6354151, -4.9378076, 3.0718241, -7.0077510, 7.5732226
7: -3.0285966, 3.0167100, -3.7234440, 3.7070456, -6.7356424, 6.7401543
8: -4.3410654, 2.4173012, -5.4249887, 2.9019449, -7.2430105, 7.8422899
9: -2.9352677, 2.9538395, -3.6089909, 3.6197283, -6.5549960, 6.5628304

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
time: 2.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
time: 2.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1953211, 4.3345661, -3.9496896, 2.7957146, -8.9910355, 8.2842560
1: -4.3170023, 4.6164837, -2.6908400, 3.0387743, -7.3557768, 7.3073235
2: -6.1278896, 4.5010343, -3.9092083, 3.0408931, -9.1687832, 8.4102421
3: -7.3889561, 3.6548228, -4.7025776, 2.4178717, -9.8068275, 8.3574009
4: -7.7128749, 4.6105943, -4.9304714, 3.1744716, -10.8873463, 9.5410652
5: -6.4589248, 3.9231412, -4.1160855, 2.6100574, -9.0689821, 8.0392265
6: -7.5325246, 4.2447419, -4.7932444, 3.0070760, -10.5396004, 9.0379868
7: -5.5208154, 5.5095692, -3.6228197, 3.6063740, -9.1271896, 9.1323891
8: -8.2391109, 4.1719885, -5.2694817, 2.8299699, -11.0690804, 9.4414701
9: -5.3339462, 5.3843799, -3.5118060, 3.5226033, -8.8565493, 8.8961859

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0940245, upper bound: 7.0941268
time: 2.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
time: 2.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.2629774, 2.3178856, -5.4263535, 3.8158021, -7.0787792, 7.7442389
1: -2.2444396, 2.5407939, -3.7753417, 4.0917711, -6.3362107, 6.3161354
2: -3.2150350, 2.5943975, -5.3774166, 4.0094271, -7.2244620, 7.9718142
3: -3.8694959, 2.0315437, -6.4753261, 3.2234309, -7.0929270, 8.5068703
4: -4.0633783, 2.7323642, -6.7870622, 4.1365519, -8.1999302, 9.5194263
5: -3.3813291, 2.2124634, -5.6790285, 3.4779925, -6.8593216, 7.8914919
6: -3.9359269, 2.6354151, -6.6366816, 3.8040371, -7.7399640, 9.2720966
7: -3.0285966, 3.0167100, -4.8853655, 4.8844371, -7.9130335, 7.9020758
8: -4.3410654, 2.4173012, -7.2433844, 3.6982622, -8.0393276, 9.6606855
9: -2.9352677, 2.9538395, -4.7286091, 4.7575555, -7.6928234, 7.6824484

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942351, upper bound: 7.0942078
time: 1.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942351, upper bound: 7.0942078
time: 2.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1953211, 4.3345661, -5.3120017, 3.7365670, -9.9318886, 9.6465683
1: -4.3170023, 4.6164837, -3.6915796, 4.0112262, -8.3282280, 8.3080635
2: -6.1278896, 4.5010343, -5.2638588, 3.9346640, -10.0625534, 9.7648926
3: -7.3889561, 3.6548228, -6.3386798, 3.1604621, -10.5494184, 9.9935026
4: -7.7128749, 4.6105943, -6.6458616, 4.0623679, -11.7752428, 11.2564564
5: -6.4589248, 3.9231412, -5.5598159, 3.4109602, -9.8698845, 9.4829569
6: -7.5325246, 4.2447419, -6.4959188, 3.7413692, -11.2738934, 10.7406607
7: -5.5208154, 5.5095692, -4.7878680, 4.7868185, -10.3076344, 10.2974377
8: -8.2391109, 4.1719885, -7.0926528, 3.6284044, -11.8675156, 11.2646408
9: -5.3339462, 5.3843799, -4.6344328, 4.6633668, -9.9973125, 10.0188122

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0941626, upper bound: 7.0942078
time: 2.29 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942351, upper bound: 7.0942078
time: 1.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.5982270, 3.2413566, -4.0674582, 2.8774748, -7.4757018, 7.3088150
1: -3.1697054, 3.5041323, -2.7771821, 3.1218820, -6.2915874, 6.2813144
2: -4.5571985, 3.4658110, -4.0264230, 3.1179724, -7.6751709, 7.4922342
3: -5.4831147, 2.7663198, -4.8434372, 2.4828491, -7.9659638, 7.6097569
4: -5.7625318, 3.5996904, -5.0759029, 3.2510345, -9.0135660, 8.6755934
5: -4.8105569, 2.9916859, -4.2390509, 2.6791158, -7.4896727, 7.2307367
6: -5.6117258, 3.3567023, -4.9378076, 3.0718241, -8.6835499, 8.2945099
7: -4.1799908, 4.1736808, -3.7234440, 3.7070456, -7.8870363, 7.8971248
8: -6.1474762, 3.1974916, -5.4249887, 2.9019449, -9.0494213, 8.6224804
9: -4.0446091, 4.0761676, -3.6089909, 3.6197283, -7.6643372, 7.6851587

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
time: 2.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
time: 2.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.9326782, 5.5283484, -3.9496896, 2.7957146, -10.7283926, 9.4780378
1: -5.5904951, 5.8529410, -2.6908400, 3.0387743, -8.6292696, 8.5437813
2: -7.8620582, 5.6379375, -3.9092083, 3.0408931, -10.9029512, 9.5471458
3: -9.4784203, 4.6039939, -4.7025776, 2.4178717, -11.8962917, 9.3065720
4: -9.9027557, 5.7387829, -4.9304714, 3.1744716, -13.0772276, 10.6692543
5: -8.2941227, 4.9457088, -4.1160855, 2.6100574, -10.9041805, 9.0617943
6: -9.7008581, 5.2041874, -4.7932444, 3.0070760, -12.7079344, 9.9974318
7: -7.0097895, 7.0129094, -3.6228197, 3.6063740, -10.6161633, 10.6357288
8: -10.5603228, 5.1728263, -5.2694817, 2.8299699, -13.3902931, 10.4423084
9: -6.7609205, 6.8487992, -3.5118060, 3.5226033, -10.2835236, 10.3606052

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0940913, upper bound: 7.0942351
time: 2.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
time: 2.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.5982270, 3.2413566, -5.4263535, 3.8158021, -8.4140291, 8.6677103
1: -3.1697054, 3.5041323, -3.7753417, 4.0917711, -7.2614765, 7.2794743
2: -4.5571985, 3.4658110, -5.3774166, 4.0094271, -8.5666256, 8.8432274
3: -5.4831147, 2.7663198, -6.4753261, 3.2234309, -8.7065458, 9.2416458
4: -5.7625318, 3.5996904, -6.7870622, 4.1365519, -9.8990841, 10.3867531
5: -4.8105569, 2.9916859, -5.6790285, 3.4779925, -8.2885494, 8.6707144
6: -5.6117258, 3.3567023, -6.6366816, 3.8040371, -9.4157629, 9.9933834
7: -4.1799908, 4.1736808, -4.8853655, 4.8844371, -9.0644283, 9.0590458
8: -6.1474762, 3.1974916, -7.2433844, 3.6982622, -9.8457384, 10.4408760
9: -4.0446091, 4.0761676, -4.7286091, 4.7575555, -8.8021641, 8.8047771

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943627, upper bound: 7.0943763
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943627, upper bound: 7.0943763
time: 2.50 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.9326782, 5.5283484, -5.3120017, 3.7365670, -11.6692448, 10.8403502
1: -5.5904951, 5.8529410, -3.6915796, 4.0112262, -9.6017208, 9.5445204
2: -7.8620582, 5.6379375, -5.2638588, 3.9346640, -11.7967224, 10.9017963
3: -9.4784203, 4.6039939, -6.3386798, 3.1604621, -12.6388826, 10.9426737
4: -9.9027557, 5.7387829, -6.6458616, 4.0623679, -13.9651241, 12.3846445
5: -8.2941227, 4.9457088, -5.5598159, 3.4109602, -11.7050829, 10.5055246
6: -9.7008581, 5.2041874, -6.4959188, 3.7413692, -13.4422274, 11.7001057
7: -7.0097895, 7.0129094, -4.7878680, 4.7868185, -11.7966080, 11.8007774
8: -10.5603228, 5.1728263, -7.0926528, 3.6284044, -14.1887274, 12.2654791
9: -6.7609205, 6.8487992, -4.6344328, 4.6633668, -11.4242878, 11.4832325

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942892, upper bound: 7.0943763
time: 2.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943627, upper bound: 7.0943763
time: 2.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.80 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998598
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998598
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0998470, upper bound: 7.0998597
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998597
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0998959, upper bound: 7.0999381
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.1000534, upper bound: 7.1000677
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0973767, upper bound: 7.0949013
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976572, upper bound: 7.0949817
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976572, upper bound: 7.0949817
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0975070, upper bound: 7.0949817
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976572, upper bound: 7.0949817
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0974159, upper bound: 7.0949797
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976299, upper bound: 7.0949797
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0977776, upper bound: 7.0951309
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0977776, upper bound: 7.0951309
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0976372, upper bound: 7.0951309
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0977776, upper bound: 7.0951309
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0948753, upper bound: 7.0975675
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949013, upper bound: 7.0975675
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949797, upper bound: 7.0976299
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949797, upper bound: 7.0976299
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949790, upper bound: 7.0976299
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949797, upper bound: 7.0976299
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949423, upper bound: 7.0976572
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0949817, upper bound: 7.0976572
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0951156, upper bound: 7.0977791
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0951156, upper bound: 7.0977791
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0951068, upper bound: 7.0977791
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0951156, upper bound: 7.0977791
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0940245, upper bound: 7.0941268
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0941268, upper bound: 7.0941268
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942351, upper bound: 7.0942078
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942351, upper bound: 7.0942078
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0941626, upper bound: 7.0942078
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942351, upper bound: 7.0942078
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0940913, upper bound: 7.0942351
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942078, upper bound: 7.0942351
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0943627, upper bound: 7.0943763
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0943627, upper bound: 7.0943763
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0942892, upper bound: 7.0943763
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 6, lower bound: -7.0943627, upper bound: 7.0943763

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -0.8242900, 0.7759713, -1.6002612, 1.6002612
1: -0.7593619, 0.7365588, -0.7593619, 0.7365588, -1.4959207, 1.4959207
2: -0.7684594, 0.9860382, -0.7684594, 0.9860382, -1.7544976, 1.7544976
3: -0.7953699, 0.6663919, -0.7953699, 0.6663919, -1.4617618, 1.4617618
4: -0.9501143, 1.0594401, -0.9501143, 1.0594401, -2.0095544, 2.0095544
5: -0.9043361, 0.8626598, -0.9043361, 0.8626598, -1.7669959, 1.7669959
6: -0.6470271, 1.4070647, -0.6470271, 1.4070647, -2.0540919, 2.0540919
7: -0.9023290, 0.9419560, -0.9023290, 0.9419560, -1.8442850, 1.8442850
8: -0.9998023, 0.9381140, -0.9998023, 0.9381140, -1.9379163, 1.9379163
9: -0.8435787, 0.9448409, -0.8435787, 0.9448409, -1.7884196, 1.7884196

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020469
time: 2.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
time: 2.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -3.2763376, 2.0824847, -2.9067748, 4.0523090
1: -0.7593619, 0.7365588, -2.1758492, 2.4862385, -3.2456005, 2.9124079
2: -0.7684594, 0.9860382, -3.0691266, 2.5236025, -3.2920618, 4.0551648
3: -0.7953699, 0.6663919, -3.9400258, 1.9516809, -2.7470508, 4.6064177
4: -0.9501143, 1.0594401, -3.8988452, 2.6804190, -3.6305332, 4.9582853
5: -0.9043361, 0.8626598, -3.2787127, 2.1279092, -3.0322452, 4.1413727
6: -0.6470271, 1.4070647, -3.9502568, 2.3441942, -2.9912214, 5.3573217
7: -0.9023290, 0.9419560, -3.0190184, 2.9432998, -3.8456287, 3.9609745
8: -0.9998023, 0.9381140, -4.0411730, 2.2886350, -3.2884374, 4.9792871
9: -0.8435787, 0.9448409, -2.8845158, 2.8212209, -3.6647997, 3.8293567

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020469
time: 2.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
time: 13.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.0823858, 1.9789348, -0.5185293, 0.5817271, -3.6641128, 2.4974642
1: -2.0647807, 2.3445125, -0.5414395, 0.5490183, -2.6137991, 2.8859520
2: -2.8838716, 2.4034491, -0.5376617, 0.7357919, -3.6196635, 2.9411106
3: -3.6913409, 1.8494611, -0.5067167, 0.4751894, -4.1665301, 2.3561778
4: -3.6660855, 2.5529046, -0.6189181, 0.8173444, -4.4834299, 3.1718225
5: -3.0894947, 2.0284617, -0.6402555, 0.6608011, -3.7502959, 2.6687171
6: -3.6895895, 2.2640123, -0.2331491, 1.3452816, -5.0348711, 2.4971614
7: -2.8481908, 2.7840900, -0.6767536, 0.6448568, -3.4930477, 3.4608436
8: -3.7992327, 2.1838686, -0.6462724, 0.7559509, -4.5551834, 2.8301411
9: -2.7217746, 2.6722190, -0.6010147, 0.6557833, -3.3775578, 3.2732339

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0982720, upper bound: 7.0979998
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0973236, upper bound: 7.0973941
time: 2.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -0.9294243, 0.8291947, -4.1055322, 3.0119090
1: -2.1758492, 2.4862385, -0.8248745, 0.7898925, -2.9657416, 3.3111129
2: -3.0691266, 2.5236025, -0.8482082, 1.0564185, -4.1255450, 3.3718107
3: -3.9400258, 1.9516809, -0.9161217, 0.7163806, -4.6564064, 2.8678026
4: -3.8988452, 2.6804190, -1.0602047, 1.1285247, -5.0273700, 3.7406237
5: -3.2787127, 2.1279092, -0.9919099, 0.9149357, -4.1936483, 3.1198192
6: -3.9502568, 2.3441942, -0.7698785, 1.4325180, -5.3827748, 3.1140728
7: -3.0190184, 2.9432998, -0.9804024, 1.0217204, -4.0407391, 3.9237022
8: -4.0411730, 2.2886350, -1.1106445, 0.9931360, -5.0343089, 3.3992796
9: -2.8845158, 2.8212209, -0.9227316, 1.0190659, -3.9035816, 3.7439525

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998470
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0998598, upper bound: 7.0998598
time: 1.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -2.0471058, 1.4479158, -2.2722058, 2.8230772
1: -0.7593619, 0.7365588, -1.4879133, 1.6022439, -2.3616059, 2.2244720
2: -0.7684594, 0.9860382, -1.9315817, 1.7790974, -2.5475569, 2.9176199
3: -0.7953699, 0.6663919, -2.3510928, 1.3185607, -2.1139307, 3.0174847
4: -0.9501143, 1.0594401, -2.4359283, 1.8813576, -2.8314719, 3.4953685
5: -0.9043361, 0.8626598, -2.1040509, 1.5007353, -2.4050713, 2.9667106
6: -0.6470271, 1.4070647, -2.3195608, 1.8668721, -2.5138993, 3.7266254
7: -0.9023290, 0.9419560, -1.9417379, 1.9663347, -2.8686638, 2.8836939
8: -0.9998023, 0.9381140, -2.5997360, 1.6031470, -2.6029494, 3.5378499
9: -0.8435787, 0.9448409, -1.8700268, 1.9104283, -2.7540069, 2.8148677

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1025248, upper bound: 7.1022677
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1025248, upper bound: 7.1023069
time: 6.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -5.2181687, 3.6797018, -4.5039916, 5.9941401
1: -0.7593619, 0.7365588, -3.4342718, 4.0089335, -4.7682953, 4.1708307
2: -0.7684594, 0.9860382, -5.2342386, 3.8684192, -4.6368785, 6.2202768
3: -0.7953699, 0.6663919, -6.3400049, 3.1585281, -3.9538980, 7.0063968
4: -0.9501143, 1.0594401, -6.6851840, 4.0211673, -4.9712815, 7.7446241
5: -0.9043361, 0.8626598, -5.5670929, 3.3006294, -4.2049656, 6.4297528
6: -0.6470271, 1.4070647, -6.4477587, 3.6439188, -4.2909460, 7.8548231
7: -0.9023290, 0.9419560, -4.7742691, 4.7608337, -5.6631627, 5.7162251
8: -0.9998023, 0.9381140, -7.0840087, 3.5179584, -4.5177608, 8.0221224
9: -0.8435787, 0.9448409, -4.6274099, 4.6256738, -5.4692526, 5.5722508

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1025248, upper bound: 7.1022677
time: 2.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1025248, upper bound: 7.1023069
time: 3.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.0823858, 1.9789348, -1.7374715, 1.2632742, -4.3456602, 3.7164063
1: -2.0647807, 2.3445125, -1.2984841, 1.3659747, -3.4307554, 3.6429965
2: -2.8838716, 2.4034491, -1.6067201, 1.5783750, -4.4622464, 4.0101690
3: -3.6913409, 1.8494611, -1.9501321, 1.1469277, -4.8382688, 3.7995932
4: -3.6660855, 2.5529046, -2.0468826, 1.6792597, -5.3453450, 4.5997872
5: -3.0894947, 2.0284617, -1.7834711, 1.3351374, -4.4246321, 3.8119328
6: -3.6895895, 2.2640123, -1.8988396, 1.7372036, -5.4267931, 4.1628518
7: -2.8481908, 2.7840900, -1.6623194, 1.6884409, -4.5366316, 4.4464092
8: -3.7992327, 2.1838686, -2.1570187, 1.4319222, -5.2311549, 4.3408871
9: -2.7217746, 2.6722190, -1.5982850, 1.6482620, -4.3700366, 4.2705040

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0982720, upper bound: 7.0979998
time: 2.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0973261, upper bound: 7.0973991
time: 3.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -2.1816595, 1.5352559, -4.8115935, 4.2641439
1: -2.1758492, 2.4862385, -1.5724896, 1.7042148, -3.8800640, 4.0587282
2: -3.0691266, 2.5236025, -2.0764785, 1.8662202, -4.9353466, 4.6000810
3: -3.9400258, 1.9516809, -2.5250976, 1.3927319, -5.3327579, 4.4767785
4: -3.8988452, 2.6804190, -2.6200173, 1.9734951, -5.8723402, 5.3004360
5: -3.2787127, 2.1279092, -2.2400005, 1.5777493, -4.8564620, 4.3679094
6: -3.9502568, 2.3441942, -2.4966009, 1.9397172, -5.8899741, 4.8407950
7: -3.0190184, 2.9432998, -2.0643587, 2.0831678, -5.1021862, 5.0076585
8: -4.0411730, 2.2886350, -2.7930062, 1.6817019, -5.7228746, 5.0816412
9: -2.8845158, 2.8212209, -1.9886416, 2.0264707, -4.9109864, 4.8098626

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0998959
time: 2.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999381, upper bound: 7.0999247
time: 2.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -0.8242900, 0.7759713, -2.8230772, 2.2722058
1: -1.4879133, 1.6022439, -0.7593619, 0.7365588, -2.2244720, 2.3616059
2: -1.9315817, 1.7790974, -0.7684594, 0.9860382, -2.9176199, 2.5475569
3: -2.3510928, 1.3185607, -0.7953699, 0.6663919, -3.0174847, 2.1139307
4: -2.4359283, 1.8813576, -0.9501143, 1.0594401, -3.4953685, 2.8314719
5: -2.1040509, 1.5007353, -0.9043361, 0.8626598, -2.9667106, 2.4050713
6: -2.3195608, 1.8668721, -0.6470271, 1.4070647, -3.7266254, 2.5138993
7: -1.9417379, 1.9663347, -0.9023290, 0.9419560, -2.8836939, 2.8686638
8: -2.5997360, 1.6031470, -0.9998023, 0.9381140, -3.5378499, 2.6029494
9: -1.8700268, 1.9104283, -0.8435787, 0.9448409, -2.8148677, 2.7540069

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020469
time: 3.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
time: 2.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -3.2763376, 2.0824847, -4.1295905, 4.7242537
1: -1.4879133, 1.6022439, -2.1758492, 2.4862385, -3.9741516, 3.7780931
2: -1.9315817, 1.7790974, -3.0691266, 2.5236025, -4.4551840, 4.8482242
3: -2.3510928, 1.3185607, -3.9400258, 1.9516809, -4.3027735, 5.2585864
4: -2.4359283, 1.8813576, -3.8988452, 2.6804190, -5.1163473, 5.7802029
5: -2.1040509, 1.5007353, -3.2787127, 2.1279092, -4.2319603, 4.7794480
6: -2.3195608, 1.8668721, -3.9502568, 2.3441942, -4.6637549, 5.8171291
7: -1.9417379, 1.9663347, -3.0190184, 2.9432998, -4.8850374, 4.9853530
8: -2.5997360, 1.6031470, -4.0411730, 2.2886350, -4.8883710, 5.6443200
9: -1.8700268, 1.9104283, -2.8845158, 2.8212209, -4.6912479, 4.7949438

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020469
time: 3.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022738, upper bound: 7.1020635
time: 2.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.0101051, 3.5310583, -0.5185293, 0.5817271, -5.5918322, 4.0495877
1: -3.3063724, 3.8497992, -0.5414395, 0.5490183, -3.8553905, 4.3912387
2: -5.0161967, 3.7299955, -0.5376617, 0.7357919, -5.7519884, 4.2676573
3: -6.0764718, 3.0378737, -0.5067167, 0.4751894, -6.5516610, 3.5445905
4: -6.4044299, 3.8793726, -0.6189181, 0.8173444, -7.2217741, 4.4982905
5: -5.3335638, 3.1828530, -0.6402555, 0.6608011, -5.9943647, 3.8231084
6: -6.1739960, 3.5279863, -0.2331491, 1.3452816, -7.5192776, 3.7611353
7: -4.5866795, 4.5747862, -0.6767536, 0.6448568, -5.2315364, 5.2515397
8: -6.7873030, 3.3937259, -0.6462724, 0.7559509, -7.5432539, 4.0399981
9: -4.4448080, 4.4453993, -0.6010147, 0.6557833, -5.1005912, 5.0464139

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0983406, upper bound: 7.0981000
time: 2.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0973276, upper bound: 7.0973942
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2181687, 3.6797018, -0.9294243, 0.8291947, -6.0473633, 4.6091261
1: -3.4342718, 4.0089335, -0.8248745, 0.7898925, -4.2241645, 4.8338079
2: -5.2342386, 3.8684192, -0.8482082, 1.0564185, -6.2906570, 4.7166271
3: -6.3400049, 3.1585281, -0.9161217, 0.7163806, -7.0563855, 4.0746498
4: -6.6851840, 4.0211673, -1.0602047, 1.1285247, -7.8137088, 5.0813723
5: -5.5670929, 3.3006294, -0.9919099, 0.9149357, -6.4820285, 4.2925391
6: -6.4477587, 3.6439188, -0.7698785, 1.4325180, -7.8802767, 4.4137974
7: -4.7742691, 4.7608337, -0.9804024, 1.0217204, -5.7959895, 5.7412362
8: -7.0840087, 3.5179584, -1.1106445, 0.9931360, -8.0771446, 4.6286030
9: -4.6274099, 4.6256738, -0.9227316, 1.0190659, -5.6464758, 5.5484052

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
time: 2.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0999247, upper bound: 7.0999381
time: 2.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -2.0471058, 1.4479158, -3.4950216, 3.4950216
1: -1.4879133, 1.6022439, -1.4879133, 1.6022439, -3.0901570, 3.0901570
2: -1.9315817, 1.7790974, -1.9315817, 1.7790974, -3.7106791, 3.7106791
3: -2.3510928, 1.3185607, -2.3510928, 1.3185607, -3.6696534, 3.6696534
4: -2.4359283, 1.8813576, -2.4359283, 1.8813576, -4.3172860, 4.3172860
5: -2.1040509, 1.5007353, -2.1040509, 1.5007353, -3.6047862, 3.6047862
6: -2.3195608, 1.8668721, -2.3195608, 1.8668721, -4.1864328, 4.1864328
7: -1.9417379, 1.9663347, -1.9417379, 1.9663347, -3.9080725, 3.9080725
8: -2.5997360, 1.6031470, -2.5997360, 1.6031470, -4.2028828, 4.2028828
9: -1.8700268, 1.9104283, -1.8700268, 1.9104283, -3.7804551, 3.7804551

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022758, upper bound: 7.1020827
time: 2.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022758, upper bound: 7.1020993
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2.0471058, 1.4479158, -5.2181687, 3.6797018, -5.7268076, 6.6660843
1: -1.4879133, 1.6022439, -3.4342718, 4.0089335, -5.4968467, 5.0365157
2: -1.9315817, 1.7790974, -5.2342386, 3.8684192, -5.8000011, 7.0133362
3: -2.3510928, 1.3185607, -6.3400049, 3.1585281, -5.5096207, 7.6585655
4: -2.4359283, 1.8813576, -6.6851840, 4.0211673, -6.4570956, 8.5665417
5: -2.1040509, 1.5007353, -5.5670929, 3.3006294, -5.4046803, 7.0678282
6: -2.3195608, 1.8668721, -6.4477587, 3.6439188, -5.9634795, 8.3146305
7: -1.9417379, 1.9663347, -4.7742691, 4.7608337, -6.7025719, 6.7406039
8: -2.5997360, 1.6031470, -7.0840087, 3.5179584, -6.1176944, 8.6871557
9: -1.8700268, 1.9104283, -4.6274099, 4.6256738, -6.4957008, 6.5378380

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022758, upper bound: 7.1020827
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1022758, upper bound: 7.1020993
time: 21.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.0101051, 3.5310583, -1.7374715, 1.2632742, -6.2733793, 5.2685299
1: -3.3063724, 3.8497992, -1.2984841, 1.3659747, -4.6723471, 5.1482830
2: -5.0161967, 3.7299955, -1.6067201, 1.5783750, -6.5945716, 5.3367157
3: -6.0764718, 3.0378737, -1.9501321, 1.1469277, -7.2233996, 4.9880056
4: -6.4044299, 3.8793726, -2.0468826, 1.6792597, -8.0836897, 5.9262552
5: -5.3335638, 3.1828530, -1.7834711, 1.3351374, -6.6687012, 4.9663239
6: -6.1739960, 3.5279863, -1.8988396, 1.7372036, -7.9111996, 5.4268260
7: -4.5866795, 4.5747862, -1.6623194, 1.6884409, -6.2751203, 6.2371054
8: -6.7873030, 3.3937259, -2.1570187, 1.4319222, -8.2192249, 5.5507445
9: -4.4448080, 4.4453993, -1.5982850, 1.6482620, -6.0930700, 6.0436840

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0983513, upper bound: 7.0981041
time: 2.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0973305, upper bound: 7.0973988
time: 3.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2181687, 3.6797018, -2.1816595, 1.5352559, -6.7534246, 5.8613615
1: -3.4342718, 4.0089335, -1.5724896, 1.7042148, -5.1384869, 5.5814233
2: -5.2342386, 3.8684192, -2.0764785, 1.8662202, -7.1004591, 5.9448977
3: -6.3400049, 3.1585281, -2.5250976, 1.3927319, -7.7327366, 5.6836257
4: -6.6851840, 4.0211673, -2.6200173, 1.9734951, -8.6586790, 6.6411848
5: -5.5670929, 3.3006294, -2.2400005, 1.5777493, -7.1448421, 5.5406299
6: -6.4477587, 3.6439188, -2.4966009, 1.9397172, -8.3874760, 6.1405196
7: -4.7742691, 4.7608337, -2.0643587, 2.0831678, -6.8574371, 6.8251925
8: -7.0840087, 3.5179584, -2.7930062, 1.6817019, -8.7657108, 6.3109646
9: -4.6274099, 4.6256738, -1.9886416, 2.0264707, -6.6538806, 6.6143155

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000631
time: 2.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000607, upper bound: 7.1000677
time: 2.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -3.2629774, 2.3178856, -3.1421757, 4.0389485
1: -0.7593619, 0.7365588, -2.2444396, 2.5407939, -3.3001559, 2.9809985
2: -0.7684594, 0.9860382, -3.2150350, 2.5943975, -3.3628569, 4.2010732
3: -0.7953699, 0.6663919, -3.8694959, 2.0315437, -2.8269136, 4.5358877
4: -0.9501143, 1.0594401, -4.0633783, 2.7323642, -3.6824784, 5.1228185
5: -0.9043361, 0.8626598, -3.3813291, 2.2124634, -3.1167994, 4.2439890
6: -0.6470271, 1.4070647, -3.9359269, 2.6354151, -3.2824421, 5.3429918
7: -0.9023290, 0.9419560, -3.0285966, 3.0167100, -3.9190390, 3.9705527
8: -0.9998023, 0.9381140, -4.3410654, 2.4173012, -3.4171035, 5.2791796
9: -0.8435787, 0.9448409, -2.9352677, 2.9538395, -3.7974181, 3.8801086

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971357
time: 2.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971582
time: 3.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -6.1953211, 4.3345661, -5.1588559, 6.9712925
1: -0.7593619, 0.7365588, -4.3170023, 4.6164837, -5.3758454, 5.0535612
2: -0.7684594, 0.9860382, -6.1278896, 4.5010343, -5.2694936, 7.1139278
3: -0.7953699, 0.6663919, -7.3889561, 3.6548228, -4.4501929, 8.0553484
4: -0.9501143, 1.0594401, -7.7128749, 4.6105943, -5.5607085, 8.7723150
5: -0.9043361, 0.8626598, -6.4589248, 3.9231412, -4.8274775, 7.3215847
6: -0.6470271, 1.4070647, -7.5325246, 4.2447419, -4.8917689, 8.9395895
7: -0.9023290, 0.9419560, -5.5208154, 5.5095692, -6.4118981, 6.4627714
8: -0.9998023, 0.9381140, -8.2391109, 4.1719885, -5.1717906, 9.1772251
9: -0.8435787, 0.9448409, -5.3339462, 5.3843799, -6.2279587, 6.2787871

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971357
time: 3.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1000064, upper bound: 7.0971582
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.0823858, 1.9789348, -2.5287571, 1.8013093, -4.8836951, 4.5076919
1: -2.0647807, 2.3445125, -1.7870578, 1.9732463, -4.0380268, 4.1315703
2: -2.8838716, 2.4034491, -2.4401972, 2.1021969, -4.9860687, 4.8436460
3: -3.6913409, 1.8494611, -2.9358237, 1.6055626, -5.2969036, 4.7852850
4: -3.6660855, 2.5529046, -3.0550196, 2.2257245, -5.8918099, 5.6079245
5: -3.0894947, 2.0284617, -2.5753431, 1.7927496, -4.8822441, 4.6038046
6: -3.6895895, 2.2640123, -2.9546392, 2.2180057, -5.9075952, 5.2186518
7: -2.8481908, 2.7840900, -2.3575201, 2.3567719, -5.2049627, 5.1416101
8: -3.7992327, 2.1838686, -3.2814662, 1.9900521, -5.7892847, 5.4653349
9: -2.7217746, 2.6722190, -2.2862234, 2.3109989, -5.0327735, 4.9584427

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0956055, upper bound: 7.0904662
time: 2.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0946648, upper bound: 7.0899577
time: 2.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.2763376, 2.0824847, -3.4376564, 2.4444370, -5.7207747, 5.5201411
1: -2.1758492, 2.4862385, -2.3536017, 2.6704078, -4.8462572, 4.8398399
2: -3.0691266, 2.5236025, -3.3958237, 2.7104023, -5.7795286, 5.9194260
3: -3.9400258, 1.9516809, -4.0846896, 2.1360202, -6.0760460, 6.0363703
4: -3.8988452, 2.6804190, -4.2867441, 2.8476155, -6.7464609, 6.9671631
5: -3.2787127, 2.1279092, -3.5712898, 2.3117449, -5.5904579, 5.6991987
6: -3.9502568, 2.3441942, -4.1573172, 2.7382574, -6.6885142, 6.5015116
7: -3.0190184, 2.9432998, -3.1834006, 3.1664240, -6.1854424, 6.1267004
8: -4.0411730, 2.2886350, -4.5822840, 2.5300474, -6.5712204, 6.8709192
9: -2.8845158, 2.8212209, -3.0857933, 3.0987611, -5.9832768, 5.9070139

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0948753
time: 2.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975675, upper bound: 7.0949013
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.8242900, 0.7759713, -4.5982270, 3.2413566, -4.0656466, 5.3741984
1: -0.7593619, 0.7365588, -3.1697054, 3.5041323, -4.2634940, 3.9062643
2: -0.7684594, 0.9860382, -4.5571985, 3.4658110, -4.2342706, 5.5432367
3: -0.7953699, 0.6663919, -5.4831147, 2.7663198, -3.5616896, 6.1495066
4: -0.9501143, 1.0594401, -5.7625318, 3.5996904, -4.5498047, 6.8219719
5: -0.9043361, 0.8626598, -4.8105569, 2.9916859, -3.8960218, 5.6732168
6: -0.6470271, 1.4070647, -5.6117258, 3.3567023, -4.0037293, 7.0187902
7: -0.9023290, 0.9419560, -4.1799908, 4.1736808, -5.0760098, 5.1219468
8: -0.9998023, 0.9381140, -6.1474762, 3.1974916, -4.1972938, 7.0855904
9: -0.8435787, 0.9448409, -4.0446091, 4.0761676, -4.9197464, 4.9894500

Time for backsubstitution: 1.26 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=7.834464073181152
rel_dist={6: [-7.107378872222906, 7.107378872222906]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1060607, upper bound: 7.1055142
time: 2.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054059, upper bound: 7.1054059
time: 3.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.06
Output dim: 6, lower bound: -7.1060607, upper bound: 7.1055142
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.06
Output dim: 6, lower bound: -7.1054059, upper bound: 7.1054059

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.3117657, 2.3313293, -3.7355754, 2.6231670, -5.9349327, 6.0669050
1: -2.2746105, 2.5741804, -2.5448871, 2.8800890, -5.1546993, 5.1190672
2: -3.2746129, 2.6151695, -3.7011802, 2.8855753, -6.1601882, 6.3163500
3: -3.9685380, 2.0463929, -4.4787531, 2.2793527, -6.2478905, 6.5251460
4: -4.1542244, 2.7427235, -4.6879644, 3.0208535, -7.1750779, 7.4306879
5: -3.4665380, 2.2262321, -3.9176166, 2.4714937, -5.9380317, 6.1438484
6: -3.9971840, 2.5632598, -4.5270967, 2.7915676, -6.7887516, 7.0903568
7: -3.0885596, 3.0902281, -3.4524512, 3.4515536, -6.5401134, 6.5426793
8: -4.4174242, 2.3710141, -4.9871321, 2.6244550, -7.0418792, 7.3581462
9: -2.9865417, 3.0139289, -3.3391511, 3.3635943, -6.3501358, 6.3530798

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1060608, upper bound: 7.1055142
time: 2.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1060608, upper bound: 7.1055142
time: 2.55 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.0443048, 4.2405229, -3.7702270, 2.6469083, -8.6912136, 8.0107498
1: -4.2253499, 4.5275450, -2.5701890, 2.9051311, -7.1304808, 7.0977340
2: -5.9989243, 4.4117823, -3.7358050, 2.9082713, -8.9071960, 8.1475868
3: -7.2230110, 3.5636163, -4.5203118, 2.2987065, -9.5217171, 8.0839281
4: -7.5507116, 4.5344086, -4.7311373, 3.0430441, -10.5937557, 9.2655458
5: -6.3241363, 3.8430071, -3.9547729, 2.4914680, -8.8156042, 7.7977800
6: -7.4014902, 4.1482449, -4.5706058, 2.8077974, -10.2092876, 8.7188511
7: -5.4129872, 5.4127798, -3.4820476, 3.4821219, -8.8951092, 8.8948269
8: -8.0652170, 4.0733213, -5.0335841, 2.6438832, -10.7091007, 9.1069050
9: -5.2316179, 5.2847004, -3.3680496, 3.3929796, -8.6245975, 8.6527500

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054059, upper bound: 7.1054059
time: 3.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054059, upper bound: 7.1054059
time: 3.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.88 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.88
Output dim: 6, lower bound: -7.1060608, upper bound: 7.1055142
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.88
Output dim: 6, lower bound: -7.1060608, upper bound: 7.1055142
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.88
Output dim: 6, lower bound: -7.1054059, upper bound: 7.1054059
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.88
Output dim: 6, lower bound: -7.1054059, upper bound: 7.1054059

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.8079071, 1.9635338, -1.7627462, 1.2832189, -4.0911260, 3.7262800
1: -1.9616743, 2.1869249, -1.3123674, 1.3856287, -3.3473029, 3.4992924
2: -2.7438402, 2.2789872, -1.6411977, 1.5972189, -4.3410592, 3.9201849
3: -3.3294473, 1.7536500, -1.9971306, 1.1623396, -4.4917870, 3.7507806
4: -3.4763777, 2.3994930, -2.0828984, 1.6768056, -5.1531835, 4.4823914
5: -2.8971171, 1.9382377, -1.8141521, 1.3481748, -4.2452917, 3.7523899
6: -3.3273859, 2.2820082, -1.9162711, 1.7241042, -5.0514898, 4.1982794
7: -2.6346295, 2.6383488, -1.6900482, 1.7088156, -4.3434448, 4.3283968
8: -3.6980345, 2.0538239, -2.1965616, 1.4503993, -5.1484337, 4.2503853
9: -2.5424128, 2.5756412, -1.6264732, 1.6750900, -4.2175026, 4.2021141

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0979057, upper bound: 7.0953982
time: 5.95 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0963841, upper bound: 7.0942275
time: 2.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.0857749, 2.1662922, -3.1502023, 2.2118421, -5.2976170, 5.3164945
1: -2.1333518, 2.4010923, -2.1738026, 2.4496696, -4.5830212, 4.5748949
2: -3.0367689, 2.4645948, -3.1049905, 2.5077229, -5.5444918, 5.5695853
3: -3.6819553, 1.9154747, -3.7644265, 1.9528916, -5.6348467, 5.6799011
4: -3.8510270, 2.5884781, -3.9378698, 2.6316509, -6.4826779, 6.5263481
5: -3.2112916, 2.0975194, -3.2829237, 2.1333621, -5.3446536, 5.3804431
6: -3.6972125, 2.4367435, -3.7796888, 2.4702182, -6.1674309, 6.2164326
7: -2.8850904, 2.8882372, -2.9446321, 2.9453170, -5.8304071, 5.8328695
8: -4.0954075, 2.2288022, -4.1864367, 2.2637742, -6.3591814, 6.4152389
9: -2.7874072, 2.8178687, -2.8446324, 2.8733115, -5.6607189, 5.6625013

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0983144, upper bound: 7.0955949
time: 4.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0966164, upper bound: 7.0944013
time: 2.42 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.4948869, 3.8601904, -1.7843635, 1.2958617, -6.7907486, 5.6445541
1: -3.8236756, 4.1367998, -1.3254743, 1.4017388, -5.2254143, 5.4622741
2: -5.4508495, 4.0511465, -1.6634818, 1.6112683, -7.0621176, 5.7146282
3: -6.5618219, 3.2623065, -2.0249252, 1.1743553, -7.7361774, 5.2872314
4: -6.8667612, 4.1783705, -2.1096969, 1.6903651, -8.5571260, 6.2880673
5: -5.7452102, 3.5187299, -1.8365524, 1.3591592, -7.1043692, 5.3552823
6: -6.7163396, 3.8425915, -1.9453349, 1.7323916, -8.4487314, 5.7879267
7: -4.9442921, 4.9405012, -1.7088956, 1.7275381, -6.6718302, 6.6493969
8: -7.3308001, 3.7382469, -2.2277317, 1.4612634, -8.7920637, 5.9659786
9: -4.7808208, 4.8225422, -1.6451657, 1.6933697, -6.4741907, 6.4677076

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0971501, upper bound: 7.0952650
time: 2.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0938241, upper bound: 7.0938356
time: 2.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.7976322, 4.0699248, -3.1795316, 2.2325196, -8.0301514, 7.2494564
1: -4.0450058, 4.3522820, -2.1918173, 2.4720726, -6.5170784, 6.5440993
2: -5.7528811, 4.2500319, -3.1356688, 2.5274518, -8.2803326, 7.3857007
3: -6.9259558, 3.4284387, -3.8013616, 1.9703177, -8.8962736, 7.2298002
4: -7.2438750, 4.3742876, -3.9768569, 2.6508131, -9.8946877, 8.3511448
5: -6.0646062, 3.6973844, -3.3165293, 2.1494250, -8.2140312, 7.0139136
6: -7.0938034, 4.0098844, -3.8183079, 2.4834664, -9.5772696, 7.8281922
7: -5.2026386, 5.2008276, -2.9711208, 2.9718957, -8.1745338, 8.1719484
8: -7.7356873, 3.9228320, -4.2284069, 2.2800238, -10.0157108, 8.1512394
9: -5.0292039, 5.0773077, -2.8706295, 2.8989987, -7.9282026, 7.9479370

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975835, upper bound: 7.0954702
time: 3.05 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0939923, upper bound: 7.0939923
time: 2.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.58 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0979057, upper bound: 7.0953982
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0963841, upper bound: 7.0942275
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0983144, upper bound: 7.0955949
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0966164, upper bound: 7.0944013
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0971501, upper bound: 7.0952650
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0938241, upper bound: 7.0938356
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0975835, upper bound: 7.0954702
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 6, lower bound: -7.0939923, upper bound: 7.0939923

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2.1155424, 1.4915136, -1.5557994, 1.1663945, -3.2819369, 3.0473130
1: -1.5302289, 1.6546773, -1.1881734, 1.2310503, -2.7612791, 2.8428507
2: -2.0042033, 1.8237524, -1.4322188, 1.4640288, -3.4682322, 3.2559712
3: -2.4381492, 1.3565404, -1.7297547, 1.0496664, -3.4878156, 3.0862951
4: -2.5294559, 1.9281955, -1.8292627, 1.5413085, -4.0707645, 3.7574582
5: -2.1734655, 1.5398579, -1.6070354, 1.2385440, -3.4120095, 3.1468933
6: -2.4113996, 1.9021864, -1.6372478, 1.6371971, -4.0485969, 3.5394342
7: -2.0031400, 2.0263166, -1.5081652, 1.5350426, -3.5381827, 3.5344820
8: -2.6981497, 1.6437011, -1.9065709, 1.3347239, -4.0328736, 3.5502720
9: -1.9300390, 1.9696834, -1.4494996, 1.5053295, -3.4353685, 3.4191830

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0977174, upper bound: 7.0953982
time: 2.74 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0979057, upper bound: 7.0953982
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2786074, 3.7231658, -1.4628295, 1.1152802, -6.3938875, 5.1859951
1: -3.4714634, 4.0552750, -1.1342374, 1.1616354, -4.6330986, 5.1895123
2: -5.2977705, 3.9087169, -1.3411114, 1.4043965, -6.7021670, 5.2498283
3: -6.4165940, 3.1934144, -1.6094304, 1.0001472, -7.4167414, 4.8028450
4: -6.7662287, 4.0625463, -1.7169911, 1.4790920, -8.2453203, 5.7795372
5: -5.6350269, 3.3350837, -1.5153296, 1.1888332, -6.8238602, 4.8504133
6: -6.5277138, 3.6778564, -1.5115920, 1.5989931, -8.1267071, 5.1894484
7: -4.8285713, 4.8146877, -1.4254194, 1.4588122, -6.2873836, 6.2401071
8: -7.1708536, 3.5556889, -1.7798334, 1.2814802, -8.4523335, 5.3355222
9: -4.6804628, 4.6783376, -1.3702096, 1.4302568, -6.1107197, 6.0485473

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0961846, upper bound: 7.0942275
time: 8.00 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0963841, upper bound: 7.0942275
time: 2.90 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2.3736458, 1.6661121, -2.9253011, 2.0485427, -4.4221888, 4.5914130
1: -1.6900173, 1.8554490, -2.0345683, 2.2791009, -3.9691181, 3.8900173
2: -2.2814879, 1.9935030, -2.8690910, 2.3585403, -4.6400280, 4.8625941
3: -2.7705262, 1.5039842, -3.4809456, 1.8211691, -4.5916953, 4.9849300
4: -2.8849473, 2.1063352, -3.6386254, 2.4806092, -5.3655567, 5.7449608
5: -2.4407649, 1.6907564, -3.0321178, 2.0060949, -4.4468598, 4.7228742
6: -2.7610040, 2.0451829, -3.4871762, 2.3459940, -5.1069980, 5.5323591
7: -2.2381377, 2.2559850, -2.7424905, 2.7458057, -4.9839435, 4.9984756
8: -3.0742040, 1.8009293, -3.8700659, 2.1248260, -5.1990299, 5.6709952
9: -2.1592188, 2.1972151, -2.6472726, 2.6803744, -4.8395929, 4.8444877

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0982311, upper bound: 7.0955949
time: 2.79 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0983143, upper bound: 7.0955949
time: 3.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.5383277, 3.9291654, -2.8322964, 1.9803421, -7.5186701, 6.7614617
1: -3.6428404, 4.2569447, -1.9764857, 2.2078156, -5.8506560, 6.2334304
2: -5.5856342, 4.0838904, -2.7698984, 2.2962101, -7.8818445, 6.8537889
3: -6.7609773, 3.3449502, -3.3619077, 1.7670561, -8.5280333, 6.7068577
4: -7.1304693, 4.2458758, -3.5130701, 2.4172506, -9.5477200, 7.7589459
5: -5.9398127, 3.4845469, -2.9280963, 1.9526463, -7.8924589, 6.4126434
6: -6.8792324, 3.8289936, -3.3659096, 2.2929654, -9.1721973, 7.1949034
7: -5.0684290, 5.0544176, -2.6574950, 2.6630878, -7.7315168, 7.7119126
8: -7.5481477, 3.7208581, -3.7370467, 2.0666618, -9.6148090, 7.4579048
9: -4.9191160, 4.9126616, -2.5648980, 2.5996366, -7.5187526, 7.4775596

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0964717, upper bound: 7.0944013
time: 3.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0966163, upper bound: 7.0944013
time: 3.18 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.6821012, 3.2989564, -1.5774198, 1.1785451, -5.8606462, 4.8763762
1: -3.2307312, 3.5638716, -1.2010186, 1.2469938, -4.4777250, 4.7648902
2: -4.6406384, 3.5207529, -1.4535046, 1.4780049, -6.1186433, 4.9742575
3: -5.5827560, 2.8126054, -1.7576673, 1.0615777, -6.6443338, 4.5702724
4: -5.8659534, 3.6545365, -1.8557584, 1.5550731, -7.4210267, 5.5102949
5: -4.8980160, 3.0411096, -1.6290543, 1.2495179, -6.1475339, 4.6701641
6: -5.7167754, 3.4027481, -1.6665570, 1.6446660, -7.3614416, 5.0693049
7: -4.2513218, 4.2456632, -1.5272837, 1.5532720, -5.8045940, 5.7729468
8: -6.2588897, 3.2480989, -1.9371734, 1.3454082, -7.6042976, 5.1852722
9: -4.1137195, 4.1461983, -1.4682934, 1.5231150, -5.6368346, 5.6144915

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0970519, upper bound: 7.0952650
time: 3.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0971501, upper bound: 7.0952650
time: 2.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.0045042, 5.5780106, -1.4827021, 1.1264646, -9.1309690, 7.0607128
1: -5.6429167, 5.9038582, -1.1457601, 1.1763303, -6.8192472, 7.0496182
2: -7.9337234, 5.6849742, -1.3605319, 1.4172653, -9.3509884, 7.0455060
3: -9.5652103, 4.6433563, -1.6350017, 1.0110015, -10.5762119, 6.2783580
4: -9.9916058, 5.7857547, -1.7413151, 1.4916745, -11.4832802, 7.5270700
5: -8.3694878, 4.9882512, -1.5356748, 1.1988339, -9.5683212, 6.5239258
6: -9.7908058, 5.2448120, -1.5382636, 1.6057062, -11.3965120, 6.7830753
7: -7.0710430, 7.0745816, -1.4429936, 1.4755698, -8.5466127, 8.5175753
8: -10.6560411, 5.2172642, -1.8075649, 1.2912327, -11.9472742, 7.0248289
9: -6.8198862, 6.9091954, -1.3873849, 1.4466891, -8.2665749, 8.2965803

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937234, upper bound: 7.0938356
time: 2.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0938241, upper bound: 7.0938356
time: 2.30 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.9733515, 3.5015683, -2.9544785, 2.0689740, -7.0423255, 6.4560471
1: -3.4437034, 3.7722344, -2.0522859, 2.3014081, -5.7451115, 5.8245201
2: -4.9328418, 3.7126868, -2.8996553, 2.3781364, -7.3109779, 6.6123419
3: -5.9332242, 2.9730737, -3.5176721, 1.8384894, -7.7717137, 6.4907455
4: -6.2273664, 3.8437088, -3.6776083, 2.4995430, -8.7269096, 7.5213170
5: -5.2060580, 3.2139003, -3.0655050, 2.0221007, -7.2281590, 6.2794056
6: -6.0808959, 3.5625768, -3.5255563, 2.3590345, -8.4399300, 7.0881329
7: -4.5006142, 4.4971194, -2.7688177, 2.7722018, -7.2728157, 7.2659369
8: -6.6494980, 3.4262691, -3.9119081, 2.1409535, -8.7904510, 7.3381772
9: -4.3536339, 4.3917303, -2.6729879, 2.7060113, -7.0596452, 7.0647182

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975593, upper bound: 7.0954702
time: 2.69 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0975835, upper bound: 7.0954702
time: 2.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.2891865, 5.7751274, -2.8598833, 1.9995581, -10.2887449, 8.6350107
1: -5.8517971, 6.1064959, -1.9932328, 2.2290294, -8.0808268, 8.0997286
2: -8.2183399, 5.8719006, -2.7988000, 2.3148623, -10.5332022, 8.6707001
3: -9.9079056, 4.7995477, -3.3966744, 1.7835529, -11.6914587, 8.1962223
4: -10.3464651, 5.9708033, -3.5499671, 2.4351554, -12.7816200, 9.5207701
5: -8.6696558, 5.1569467, -2.9597983, 1.9678196, -10.6374750, 8.1167450
6: -10.1458530, 5.4049721, -3.4023137, 2.3050299, -12.4508829, 8.8072853
7: -7.3148050, 7.3197269, -2.6824725, 2.6882110, -10.0030155, 10.0021992
8: -11.0372505, 5.3910294, -3.7768357, 2.0818224, -13.1190729, 9.1678648
9: -7.0537591, 7.1489582, -2.5893695, 2.6239419, -9.6777010, 9.7383280

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0939455, upper bound: 7.0939923
time: 4.02 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0939923, upper bound: 7.0939923
time: 1.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.16 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0977174, upper bound: 7.0953982
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0979057, upper bound: 7.0953982
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0961846, upper bound: 7.0942275
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0963841, upper bound: 7.0942275
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0982311, upper bound: 7.0955949
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0983143, upper bound: 7.0955949
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0964717, upper bound: 7.0944013
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0966163, upper bound: 7.0944013
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0970519, upper bound: 7.0952650
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0971501, upper bound: 7.0952650
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0937234, upper bound: 7.0938356
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0938241, upper bound: 7.0938356
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0975593, upper bound: 7.0954702
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0975835, upper bound: 7.0954702
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0939455, upper bound: 7.0939923
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.16
Output dim: 6, lower bound: -7.0939923, upper bound: 7.0939923

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.5720320, 1.1733315, -0.6241183, 0.6515307, -2.2235627, 1.7974498
1: -1.1963099, 1.2459762, -0.6243891, 0.6149392, -1.8112490, 1.8703653
2: -1.4398944, 1.4754804, -0.6159981, 0.8190569, -2.2589512, 2.0914786
3: -1.7384530, 1.0615212, -0.5935259, 0.5432796, -2.2817326, 1.6550472
4: -1.8439409, 1.5671299, -0.7282928, 0.9075595, -2.7515004, 2.2954226
5: -1.6235535, 1.2487154, -0.7274276, 0.7304064, -2.3539600, 1.9761430
6: -1.6818073, 1.6501203, -0.3743296, 1.3712237, -3.0530310, 2.0244498
7: -1.5175891, 1.5547519, -0.7482324, 0.7470668, -2.2646558, 2.3029842
8: -1.9293822, 1.3388816, -0.7559971, 0.8301353, -2.7595174, 2.0948787
9: -1.4598311, 1.5169859, -0.6849291, 0.7590286, -2.2188597, 2.2019150

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 4.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 3.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.0065608, 1.4223746, -1.1330388, 0.9363815, -2.9429424, 2.5554132
1: -1.4626418, 1.5711207, -0.9468466, 0.9189075, -2.3815494, 2.5179672
2: -1.8884796, 1.7526059, -1.0250835, 1.1932955, -3.0817752, 2.7776895
3: -2.2985158, 1.2958093, -1.1797549, 0.8213792, -3.1198950, 2.4755640
4: -2.3803234, 1.8542769, -1.3129350, 1.2615862, -3.6419096, 3.1672120
5: -2.0623789, 1.4779646, -1.1876237, 1.0196979, -3.0820768, 2.6655884
6: -2.2656624, 1.8475882, -1.0536861, 1.4903040, -3.7559664, 2.9012742
7: -1.9045317, 1.9296885, -1.1475475, 1.1827780, -3.0873098, 3.0772359
8: -2.5427101, 1.5807405, -1.3601811, 1.1065512, -3.6492612, 2.9409215
9: -1.8340820, 1.8755667, -1.0937972, 1.1729336, -3.0070157, 2.9693639

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0959462, upper bound: 7.0903147
time: 3.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947559, upper bound: 7.0898348
time: 2.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.7046528, 3.3131039, -0.5857565, 0.6266414, -5.3312941, 3.8988605
1: -3.1186328, 3.6162353, -0.5947561, 0.5911629, -3.7097957, 4.2109914
2: -4.6962881, 3.5268440, -0.5878335, 0.7889423, -5.4852304, 4.1146774
3: -5.6896305, 2.8605697, -0.5602591, 0.5190035, -6.2086339, 3.4208288
4: -5.9917603, 3.6714234, -0.6881378, 0.8757485, -6.8675089, 4.3595610
5: -4.9907441, 3.0101731, -0.6964169, 0.7051151, -5.6958590, 3.7065899
6: -5.7725143, 3.3581529, -0.3240589, 1.3612311, -7.1337452, 3.6822119
7: -4.3110762, 4.3013988, -0.7227997, 0.7097170, -5.0207930, 5.0241985
8: -6.3522916, 3.2130883, -0.7158241, 0.8035278, -7.1558194, 3.9289124
9: -4.1767387, 4.1810293, -0.6554782, 0.7221515, -4.8988905, 4.8365073

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937021, upper bound: 7.0892591
time: 2.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
time: 2.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1652632, 3.6420877, -1.0477312, 0.8909262, -6.0561895, 4.6898189
1: -3.4018822, 3.9684427, -0.8954763, 0.8624271, -4.2643094, 4.8639193
2: -5.1789021, 3.8332651, -0.9470852, 1.1365273, -6.3154297, 4.7803502
3: -6.2729535, 3.1275973, -1.0670717, 0.7762510, -7.0492043, 4.1946688
4: -6.6133537, 3.9854462, -1.2054139, 1.2057750, -7.8191290, 5.1908603
5: -5.5075259, 3.2709153, -1.1046916, 0.9752488, -6.4827747, 4.3756070
6: -6.3785191, 3.6150987, -0.9332347, 1.4638634, -7.8423824, 4.5483332
7: -4.7263942, 4.7131243, -1.0779862, 1.1143243, -5.8407183, 5.7911105
8: -7.0089531, 3.4878292, -1.2533795, 1.0576319, -8.0665846, 4.7412086
9: -4.5808916, 4.5799513, -1.0216218, 1.1069336, -5.6878252, 5.6015730

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943064, upper bound: 7.0892599
time: 2.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0931676, upper bound: 7.0889357
time: 2.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.8249135, 1.3161880, -2.0089424, 1.4229819, -3.2478952, 3.3251305
1: -1.3495984, 1.4365354, -1.4661871, 1.5693650, -2.9189634, 2.9027224
2: -1.6982026, 1.6373419, -1.8887839, 1.7523154, -3.4505181, 3.5261259
3: -2.0629025, 1.1974868, -2.2986338, 1.2937402, -3.3566427, 3.4961205
4: -2.1545808, 1.7338384, -2.3809712, 1.8600554, -4.0146360, 4.1148095
5: -1.8783573, 1.3825147, -2.0587873, 1.4799167, -3.3582740, 3.4413021
6: -2.0272102, 1.7647104, -2.2651882, 1.8651826, -3.8923929, 4.0298986
7: -1.7381657, 1.7707880, -1.9040146, 1.9237180, -3.6618838, 3.6748025
8: -2.2864046, 1.4868276, -2.5416620, 1.5894815, -3.8758860, 4.0284896
9: -1.6755875, 1.7254524, -1.8320500, 1.8740977, -3.5496852, 3.5575023

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0958112, upper bound: 7.0903147
time: 3.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947962, upper bound: 7.0898348
time: 4.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2.2622330, 1.5913503, -2.4509568, 1.7171390, -3.9793720, 4.0423069
1: -1.6207993, 1.7693928, -1.7408999, 1.9120579, -3.5328572, 3.5102928
2: -2.1622159, 1.9204112, -2.3648491, 2.0434222, -4.2056379, 4.2852602
3: -2.6261492, 1.4401422, -2.8725557, 1.5459690, -4.1721182, 4.3126979
4: -2.7308075, 2.0303817, -2.9918280, 2.1590321, -4.8898396, 5.0222096
5: -2.3252370, 1.6264479, -2.5186369, 1.7340199, -4.0592570, 4.1450849
6: -2.6123471, 1.9854624, -2.8586583, 2.0886350, -4.7009821, 4.8441210
7: -2.1360140, 2.1569710, -2.3105485, 2.3214431, -4.4574571, 4.4675198
8: -2.9121885, 1.7363063, -3.1851990, 1.8427308, -4.7549191, 4.9215055
9: -2.0604277, 2.0987034, -2.2270763, 2.2644794, -4.3249073, 4.3257799

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0960948, upper bound: 7.0903147
time: 2.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949635, upper bound: 7.0898348
time: 3.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.9678888, 3.5168791, -1.9180546, 1.3670189, -6.3349075, 5.4349337
1: -3.2894373, 3.8188953, -1.4092529, 1.5014057, -4.7908430, 5.2281485
2: -4.9834881, 3.7034833, -1.7923908, 1.6942797, -6.6777678, 5.4958744
3: -6.0343800, 3.0131340, -2.1819468, 1.2436888, -7.2780685, 5.1950808
4: -6.3572216, 3.8560076, -2.2697442, 1.7990679, -8.1562891, 6.1257515
5: -5.2956557, 3.1606548, -1.9674386, 1.4305856, -6.7262411, 5.1280932
6: -6.1255875, 3.5103462, -2.1459005, 1.8196582, -7.9452457, 5.6562467
7: -4.5518875, 4.5409360, -1.8217845, 1.8449644, -6.3968520, 6.3627205
8: -6.7322216, 3.3798585, -2.4142625, 1.5353038, -8.2675257, 5.7941208
9: -4.4149537, 4.4158993, -1.7523059, 1.7993175, -6.2142711, 6.1682053

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937861, upper bound: 7.0892591
time: 2.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0928197, upper bound: 7.0889357
time: 4.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.4231520, 3.8456163, -2.3598595, 1.6543239, -7.0774760, 6.2054758
1: -3.5715263, 4.1678824, -1.6830890, 1.8421326, -5.4136591, 5.8509712
2: -5.4637394, 4.0067368, -2.2661371, 1.9832931, -7.4470325, 6.2728739
3: -6.6140757, 3.2775140, -2.7535539, 1.4943105, -8.1083860, 6.0310678
4: -6.9743171, 4.1672444, -2.8655221, 2.0960262, -9.0703430, 7.0327663
5: -5.8092051, 3.4189241, -2.4244130, 1.6811159, -7.4903212, 5.8433371
6: -6.7264462, 3.7650318, -2.7381978, 2.0364373, -8.7628832, 6.5032296
7: -4.9639788, 4.9499884, -2.2262831, 2.2411466, -7.2051253, 7.1762714
8: -7.3827724, 3.6517882, -3.0525911, 1.7877815, -9.1705542, 6.7043791
9: -4.8167586, 4.8119230, -2.1460371, 2.1838522, -7.0006108, 6.9579601

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943373, upper bound: 7.0892599
time: 2.21 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0931771, upper bound: 7.0889357
time: 3.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.0464625, 2.8605263, -0.6367497, 0.6607352, -4.7071977, 3.4972761
1: -2.7636447, 3.1112206, -0.6343385, 0.6230162, -3.3866611, 3.7455592
2: -4.0035539, 3.1046851, -0.6255338, 0.8294887, -4.8330426, 3.7302189
3: -4.8180552, 2.4643879, -0.6047674, 0.5519288, -5.3699837, 3.0691552
4: -5.0698690, 3.2449682, -0.7427117, 0.9170018, -5.9868708, 3.9876800
5: -4.2282572, 2.6672230, -0.7380759, 0.7394998, -4.9677572, 3.4052987
6: -4.9317203, 3.0619092, -0.3903074, 1.3731638, -6.3048840, 3.4522166
7: -3.7047322, 3.6976180, -0.7568071, 0.7600501, -4.4647822, 4.4544253
8: -5.4105306, 2.8753400, -0.7703115, 0.8375725, -6.2481031, 3.6456516
9: -3.5900071, 3.6124117, -0.6952150, 0.7721860, -4.3621931, 4.3076267

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947621, upper bound: 7.0901832
time: 3.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0928529, upper bound: 7.0896393
time: 2.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.5585356, 3.2138245, -1.1541662, 0.9475710, -5.5061064, 4.3679905
1: -3.1401079, 3.4757690, -0.9591240, 0.9329545, -4.0730624, 4.4348931
2: -4.5170565, 3.4398561, -1.0443140, 1.2070012, -5.7240577, 4.4841700
3: -5.4342480, 2.7446876, -1.2072831, 0.8327850, -6.2670331, 3.9519706
4: -5.7120066, 3.5749397, -1.3391236, 1.2751441, -6.9871507, 4.9140635
5: -4.7678633, 2.9684873, -1.2086980, 1.0298588, -5.7977219, 4.1771851
6: -5.5639124, 3.3372402, -1.0831599, 1.4954809, -7.0593934, 4.4204001
7: -4.1454430, 4.1387820, -1.1641648, 1.1997347, -5.3451777, 5.3029470
8: -6.0943193, 3.1754265, -1.3864086, 1.1168828, -7.2112021, 4.5618353
9: -4.0117617, 4.0423617, -1.1114371, 1.1895419, -5.2013035, 5.1537991

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0950684, upper bound: 7.0901832
time: 3.47 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0930090, upper bound: 7.0896415
time: 3.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3697410, 5.1407819, -0.5966620, 0.6348653, -8.0046062, 5.7374439
1: -5.1775379, 5.4532833, -0.6040359, 0.5981814, -5.7757192, 6.0573192
2: -7.2984958, 5.2696514, -0.5959913, 0.7979567, -8.0964527, 5.8656425
3: -8.7962952, 4.2961597, -0.5689889, 0.5267478, -9.3230429, 4.8651485
4: -9.1956635, 5.3748627, -0.6998587, 0.8844514, -10.0801144, 6.0747213
5: -7.7007742, 4.6147246, -0.7059262, 0.7123523, -8.4131260, 5.3206511
6: -9.0035810, 4.8929272, -0.3374181, 1.3628855, -10.3664665, 5.2303452
7: -6.5257645, 6.5279298, -0.7299776, 0.7210940, -7.2468586, 7.2579074
8: -9.8098698, 4.8443751, -0.7277887, 0.8101176, -10.6199875, 5.5721641
9: -6.2979045, 6.3747983, -0.6647843, 0.7334474, -7.0313520, 7.0395827

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907834, upper bound: 7.0888620
time: 2.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0884682, upper bound: 7.0885707
time: 2.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.8812718, 5.4931684, -1.0663439, 0.9010381, -8.7823095, 6.5595121
1: -5.5526018, 5.8164330, -0.9067076, 0.8744177, -6.4270196, 6.7231407
2: -7.8104973, 5.6044044, -0.9639270, 1.1491551, -8.9596519, 6.5683317
3: -9.4161377, 4.5758939, -1.0913904, 0.7861780, -10.2023153, 5.6672840
4: -9.8374624, 5.7061973, -1.2295575, 1.2170167, -11.0544796, 6.9357548
5: -8.2395649, 4.9158177, -1.1231446, 0.9845859, -9.2241507, 6.0389624
6: -9.6381407, 5.1770229, -0.9584810, 1.4684153, -11.1065559, 6.1355038
7: -6.9653373, 6.9683237, -1.0930898, 1.1294587, -8.0947962, 8.0614138
8: -10.4918594, 5.1446795, -1.2773302, 1.0666512, -11.5585108, 6.4220095
9: -6.7185292, 6.8054519, -1.0374148, 1.1218033, -7.8403325, 7.8428669

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888619
time: 2.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885593, upper bound: 7.0885707
time: 2.51 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.3342543, 3.0609803, -2.0370975, 1.4407682, -5.7750225, 5.0980778
1: -2.9751678, 3.3172443, -1.4834076, 1.5910461, -4.5662136, 4.8006520
2: -4.2923379, 3.2940774, -1.9186178, 1.7710255, -6.0633636, 5.2126951
3: -5.1627612, 2.6224830, -2.3345675, 1.3099087, -6.4726701, 4.9570503
4: -5.4289846, 3.4300218, -2.4194505, 1.8782012, -7.3071861, 5.8494720
5: -4.5327549, 2.8379359, -2.0881777, 1.4954193, -6.0281744, 4.9261136
6: -5.2896175, 3.2219200, -2.3028584, 1.8767877, -7.1664052, 5.5247784
7: -3.9514589, 3.9457047, -1.9294624, 1.9491073, -5.9005661, 5.8751669
8: -5.7978964, 3.0520689, -2.5821817, 1.6044512, -7.4023476, 5.6342506
9: -3.8267903, 3.8547106, -1.8571832, 1.8988594, -5.7256498, 5.7118940

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951315, upper bound: 7.0901832
time: 2.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0932615, upper bound: 7.0896393
time: 2.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8474240, 3.4147332, -2.4814439, 1.7371244, -6.5845485, 5.8961773
1: -3.3514712, 3.6826539, -1.7597437, 1.9356358, -5.2871070, 5.4423976
2: -4.8067670, 3.6302025, -2.3973417, 2.0636330, -6.8704000, 6.0275440
3: -5.7809196, 2.9039936, -2.9118299, 1.5637376, -7.3446569, 5.8158236
4: -6.0702472, 3.7624855, -3.0338311, 2.1791651, -8.2494125, 6.7963166
5: -5.0732226, 3.1398664, -2.5509067, 1.7509866, -6.8242092, 5.6907730
6: -5.9247570, 3.4960032, -2.8992920, 2.1025059, -8.0272627, 6.3952951
7: -4.3925648, 4.3884211, -2.3384471, 2.3487940, -6.7413588, 6.7268682
8: -6.4815793, 3.3522053, -3.2300892, 1.8590146, -8.3405943, 6.5822945
9: -4.2498703, 4.2857828, -2.2541881, 2.2917798, -6.5416498, 6.5399709

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953006, upper bound: 7.0901832
time: 3.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0933477, upper bound: 7.0896415
time: 2.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.6574287, 5.3401423, -1.9448112, 1.3830862, -9.0405149, 7.2849536
1: -5.3882465, 5.6580257, -1.4255061, 1.5216668, -6.9099131, 7.0835319
2: -7.5862985, 5.4588094, -1.8204595, 1.7117169, -9.2980156, 7.2792687
3: -9.1431093, 4.4537449, -2.2162325, 1.2588842, -10.4019938, 6.6699772
4: -9.5553112, 5.5619564, -2.3025711, 1.8159626, -11.3712740, 7.8645277
5: -8.0043192, 4.7849188, -1.9952540, 1.4445536, -9.4488726, 6.7801728
6: -9.3620853, 5.0552301, -2.1816115, 1.8298458, -11.1919308, 7.2368417
7: -6.7720027, 6.7749233, -1.8459839, 1.8687978, -8.6408005, 8.6209068
8: -10.1959286, 5.0199318, -2.4529448, 1.5490854, -11.7450142, 7.4728765
9: -6.5339489, 6.6174183, -1.7760993, 1.8220834, -8.3560324, 8.3935175

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907856, upper bound: 7.0888620
time: 2.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885028, upper bound: 7.0885707
time: 4.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1641083, 5.6888690, -2.3889070, 1.6731663, -9.8372746, 8.0777760
1: -5.7597733, 6.0173278, -1.7010477, 1.8644150, -7.6241884, 7.7183752
2: -8.0930109, 5.7901349, -2.2969699, 2.0025127, -10.0955238, 8.0871048
3: -9.7566643, 4.7309480, -2.7908707, 1.5112003, -11.2678642, 7.5218186
4: -10.1900520, 5.8900867, -2.9054689, 2.1150677, -12.3051195, 8.7955551
5: -8.5375490, 5.0829391, -2.4549656, 1.6972508, -10.2348003, 7.5379047
6: -9.9903431, 5.3361716, -2.7768562, 2.0492411, -12.0395842, 8.1130276
7: -7.2071257, 7.2112303, -2.2528443, 2.2671962, -9.4743214, 9.4640751
8: -10.8703146, 5.3174376, -3.0950594, 1.8031406, -12.6734552, 8.4124966
9: -6.9506111, 7.0435491, -2.1718011, 2.2099204, -9.1605320, 9.2153502

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
time: 2.26 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885707, upper bound: 7.0885707
time: 2.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.53 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0959462, upper bound: 7.0903147
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0947559, upper bound: 7.0898348
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0937021, upper bound: 7.0892591
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0943064, upper bound: 7.0892599
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0931676, upper bound: 7.0889357
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0958112, upper bound: 7.0903147
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0947962, upper bound: 7.0898348
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0960948, upper bound: 7.0903147
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0949635, upper bound: 7.0898348
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0937861, upper bound: 7.0892591
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0928197, upper bound: 7.0889357
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0943373, upper bound: 7.0892599
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0931771, upper bound: 7.0889357
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0947621, upper bound: 7.0901832
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0928529, upper bound: 7.0896393
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0950684, upper bound: 7.0901832
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0930090, upper bound: 7.0896415
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0907834, upper bound: 7.0888620
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0884682, upper bound: 7.0885707
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888619
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0885593, upper bound: 7.0885707
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0951315, upper bound: 7.0901832
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0932615, upper bound: 7.0896393
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0953006, upper bound: 7.0901832
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0933477, upper bound: 7.0896415
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0907856, upper bound: 7.0888620
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0885028, upper bound: 7.0885707
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.53
Output dim: 6, lower bound: -7.0885707, upper bound: 7.0885707

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9007107, 0.8166325, -0.5188838, 0.5815122, -1.4822229, 1.3355162
1: -0.8046172, 0.7802006, -0.5418224, 0.5495160, -1.3541331, 1.3220229
2: -0.8202603, 1.0439324, -0.5382037, 0.7360134, -1.5562737, 1.5821362
3: -0.8737449, 0.7098458, -0.5068530, 0.4753516, -1.3490965, 1.2166988
4: -1.0246694, 1.1293843, -0.6188372, 0.8184482, -1.8431177, 1.7482215
5: -0.9702371, 0.9062064, -0.6402802, 0.6613756, -1.6316128, 1.5464866
6: -0.7667005, 1.4258229, -0.2366513, 1.3469745, -2.1136751, 1.6624742
7: -0.9592206, 1.0084324, -0.6771295, 0.6446044, -1.6038251, 1.6855619
8: -1.0847286, 0.9787750, -0.6462300, 0.7600299, -1.8447585, 1.6250050
9: -0.9018761, 1.0038450, -0.6017593, 0.6550869, -1.5569630, 1.6056043

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 4.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -2.7110963, 1.7823443, -0.5086367, 0.5735273, -3.2846236, 2.2909811
1: -1.8644289, 2.0951281, -0.5331682, 0.5429302, -2.4073591, 2.6282964
2: -2.5329409, 2.1978054, -0.5295452, 0.7279655, -3.2609062, 2.7273507
3: -3.1576500, 1.6583755, -0.4989797, 0.4692504, -3.6269004, 2.1573553
4: -3.1931124, 2.3768959, -0.6072972, 0.8096800, -4.0027924, 2.9841931
5: -2.7069514, 1.8717282, -0.6311710, 0.6540871, -3.3610384, 2.5028992
6: -3.2545156, 2.2633750, -0.2247546, 1.3452927, -4.5998082, 2.4881296
7: -2.5033851, 2.4735458, -0.6692647, 0.6342808, -3.1376657, 3.1428106
8: -3.4588385, 2.0934308, -0.6356186, 0.7531743, -4.2120128, 2.7290494
9: -2.4083297, 2.4128132, -0.5936167, 0.6428906, -3.0512204, 3.0064299

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 3.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.2656287, 1.0083714, -0.9173641, 0.8222614, -2.0878901, 1.9257355
1: -1.0195036, 1.0204811, -0.8177124, 0.7838798, -1.8033834, 1.8381934
2: -1.1430192, 1.2835000, -0.8398364, 1.0470345, -2.1900537, 2.1233363
3: -1.3459476, 0.8986995, -0.9016583, 0.7100151, -2.0559626, 1.8003578
4: -1.4721084, 1.3677002, -1.0467994, 1.1220607, -2.5941691, 2.4144998
5: -1.3211970, 1.0928774, -0.9813139, 0.9090835, -2.2302806, 2.0741913
6: -1.2669408, 1.5314304, -0.7580701, 1.4322640, -2.6992049, 2.2895005
7: -1.2533193, 1.3006904, -0.9716480, 1.0109743, -2.2642937, 2.2723384
8: -1.5347754, 1.1699867, -1.0970610, 0.9917845, -2.5265598, 2.2670479
9: -1.2029952, 1.2798890, -0.9137685, 1.0094814, -2.2124767, 2.1936574

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0959462, upper bound: 7.0903147
time: 3.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0959462, upper bound: 7.0903147
time: 2.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.2414737, 2.1565280, -0.8936258, 0.8096340, -4.0511079, 3.0501537
1: -2.2303569, 2.5118790, -0.8033419, 0.7716439, -3.0020008, 3.3152208
2: -3.1797576, 2.5396690, -0.8220834, 1.0304446, -4.2102022, 3.3617525
3: -3.8472333, 1.9607602, -0.8737833, 0.6984951, -4.5457287, 2.8345437
4: -3.8699775, 2.7453194, -1.0213389, 1.1066591, -4.9766364, 3.7666583
5: -3.2844045, 2.1705277, -0.9605913, 0.8976578, -4.1820621, 3.1311190
6: -3.9639344, 2.5611429, -0.7300940, 1.4280522, -5.3919868, 3.2912369
7: -3.0012994, 2.9981453, -0.9543146, 0.9923323, -3.9936318, 3.9524598
8: -4.2727776, 2.4142337, -1.0718554, 0.9806953, -5.2534728, 3.4860892
9: -2.9014208, 2.8849616, -0.8953518, 0.9924630, -3.8938837, 3.7803135

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947559, upper bound: 7.0898348
time: 2.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947559, upper bound: 7.0898348
time: 2.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.9496837, 2.7732000, -0.4875644, 0.5593813, -4.5090652, 3.2607644
1: -2.6545053, 3.0374718, -0.5151396, 0.5299978, -3.1845031, 3.5526114
2: -3.9045963, 3.0238886, -0.5114501, 0.7124265, -4.6170230, 3.5353386
3: -4.7330346, 2.4226484, -0.4832263, 0.4579863, -5.1910210, 2.9058747
4: -4.9728127, 3.1564448, -0.5843949, 0.7907275, -5.7635403, 3.7408397
5: -4.1421041, 2.5822048, -0.6132923, 0.6390241, -4.7811284, 3.1954970
6: -4.7790027, 2.9381104, -0.1965496, 1.3376248, -6.1166277, 3.1346600
7: -3.6299500, 3.6251717, -0.6531762, 0.6147484, -4.2446985, 4.2783480
8: -5.2738137, 2.7618079, -0.6140943, 0.7347225, -6.0085363, 3.3759022
9: -3.5135031, 3.5259776, -0.5773752, 0.6215600, -4.1350632, 4.1033525

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937021, upper bound: 7.0892591
time: 2.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937021, upper bound: 7.0892591
time: 2.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2343907, 4.4305501, -0.4789623, 0.5525493, -6.7869401, 4.9095125
1: -4.0474377, 4.7808418, -0.5075518, 0.5246691, -4.5721068, 5.2883935
2: -6.2693372, 4.5503922, -0.5044018, 0.7053396, -6.9746766, 5.0547938
3: -7.5463839, 3.7759576, -0.4771190, 0.4527846, -7.9991684, 4.2530766
4: -8.0034208, 4.7335567, -0.5744387, 0.7836486, -8.7870693, 5.3079953
5: -6.6393270, 3.9045503, -0.6053420, 0.6328013, -7.2721281, 4.5098925
6: -7.7710896, 4.3196278, -0.1865726, 1.3364749, -9.1075649, 4.5062003
7: -5.6423306, 5.6192970, -0.6467599, 0.6056013, -6.2479320, 6.2660570
8: -8.4927101, 4.2740993, -0.6049693, 0.7293177, -9.2220278, 4.8790689
9: -5.4949799, 5.4606066, -0.5704302, 0.6111664, -6.1061463, 6.0310369

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
time: 2.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.4011250, 3.0958078, -0.8423291, 0.7841936, -5.1853185, 3.9381371
1: -2.9321208, 3.3827810, -0.7708067, 0.7454952, -3.6776161, 4.1535878
2: -4.3777075, 3.3242993, -0.7832617, 0.9966987, -5.3744063, 4.1075611
3: -5.3047566, 2.6843410, -0.8160936, 0.6739191, -5.9786758, 3.5004344
4: -5.5820398, 3.4644499, -0.9690488, 1.0713524, -6.6533923, 4.4334989
5: -4.6485653, 2.8379536, -0.9183916, 0.8717437, -5.5203090, 3.7563453
6: -5.3704548, 3.1902380, -0.6686218, 1.4146781, -6.7851329, 3.8588598
7: -4.0371370, 4.0287180, -0.9161007, 0.9537296, -4.9908667, 4.9448185
8: -5.9174786, 3.0312867, -1.0188922, 0.9502306, -6.8677092, 4.0501790
9: -3.9097035, 3.9169817, -0.8573632, 0.9565664, -4.8662701, 4.7743449

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943064, upper bound: 7.0892599
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943064, upper bound: 7.0892599
time: 2.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.6910205, 4.7560925, -0.8229812, 0.7741511, -7.4651718, 5.5790739
1: -4.3279586, 5.1300368, -0.7592580, 0.7360335, -5.0639920, 5.8892946
2: -6.7480650, 4.8546081, -0.7694943, 0.9829890, -7.7310538, 5.6241026
3: -8.1249905, 4.0409760, -0.7955655, 0.6639663, -8.7889566, 4.8365417
4: -8.6203232, 5.0448637, -0.9492531, 1.0593944, -9.6797180, 5.9941168
5: -7.1515841, 4.1631036, -0.9023840, 0.8618101, -8.0133944, 5.0654879
6: -8.3719692, 4.5666485, -0.6463437, 1.4118361, -9.7838058, 5.2129922
7: -6.0535264, 6.0281744, -0.9020419, 0.9384282, -6.9919548, 6.9302163
8: -9.1442451, 4.5454135, -0.9985768, 0.9415716, -10.0858164, 5.5439901
9: -5.8960967, 5.8566270, -0.8429712, 0.9419024, -6.8379993, 6.6995983

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0931676, upper bound: 7.0889357
time: 2.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0931676, upper bound: 7.0889357
time: 2.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.1101272, 0.9252712, -1.7433189, 1.2670627, -2.3771899, 2.6685901
1: -0.9288674, 0.9105242, -1.3020216, 1.3713790, -2.3002465, 2.2125459
2: -0.9973681, 1.1835456, -1.6135757, 1.5826225, -2.5799906, 2.7971213
3: -1.1414398, 0.8160803, -1.9577014, 1.1495434, -2.2909832, 2.7737818
4: -1.2784632, 1.2653136, -2.0537181, 1.6840841, -2.9625473, 3.3190317
5: -1.1683387, 1.0147225, -1.7894593, 1.3391976, -2.5075364, 2.8041818
6: -1.0554105, 1.4840273, -1.9096819, 1.7419695, -2.7973800, 3.3937092
7: -1.1269107, 1.1732323, -1.6675209, 1.6931987, -2.8201094, 2.8407531
8: -1.3378910, 1.0962948, -2.1670737, 1.4394611, -2.7773521, 3.2633686
9: -1.0741626, 1.1586128, -1.6037881, 1.6540164, -2.7281790, 2.7624011

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0958112, upper bound: 7.0903147
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0958112, upper bound: 7.0903147
time: 3.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -2.9713013, 1.9627411, -1.7114818, 1.2487612, -4.2200623, 3.6742229
1: -2.0561345, 2.2932339, -1.2828372, 1.3473577, -3.4034922, 3.5760710
2: -2.8653095, 2.3667958, -1.5808249, 1.5620834, -4.4273930, 3.9476206
3: -3.5020635, 1.7947205, -1.9164368, 1.1321498, -4.6342134, 3.7111573
4: -3.5363147, 2.5558853, -2.0141168, 1.6634347, -5.1997495, 4.5700021
5: -2.9972427, 2.0111067, -1.7569594, 1.3226416, -4.3198843, 3.7680662
6: -3.6127625, 2.4113882, -1.8667364, 1.7289060, -5.3416686, 4.2781248
7: -2.7354150, 2.7369463, -1.6395698, 1.6656518, -4.4010668, 4.3765163
8: -3.8852530, 2.2604086, -2.1215696, 1.4225869, -5.3078399, 4.3819780
9: -2.6295478, 2.6657252, -1.5763071, 1.6273847, -4.2569323, 4.2420321

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947962, upper bound: 7.0898348
time: 2.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947962, upper bound: 7.0898348
time: 3.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.5100877, 1.1390880, -2.1814141, 1.5361133, -3.0462010, 3.3205023
1: -1.1598783, 1.2006688, -1.5728045, 1.7033006, -2.8631787, 2.7734733
2: -1.3791571, 1.4357743, -2.0758810, 1.8655962, -3.2447534, 3.5116553
3: -1.6567867, 1.0276104, -2.5240035, 1.3912593, -3.0480461, 3.5516138
4: -1.7665523, 1.5282332, -2.6200845, 1.9746163, -3.7411685, 4.1483178
5: -1.5607721, 1.2179480, -2.2392988, 1.5779006, -3.1386728, 3.4572468
6: -1.6018507, 1.6294168, -2.4970164, 1.9431922, -3.5450430, 4.1264334
7: -1.4615996, 1.5024477, -2.0631745, 2.0805528, -3.5421524, 3.5656223
8: -1.8443092, 1.3130963, -2.7925911, 1.6867948, -3.5311041, 4.1056871
9: -1.4060476, 1.4665107, -1.9873084, 2.0261769, -3.4322245, 3.4538190

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0960948, upper bound: 7.0903147
time: 2.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0960948, upper bound: 7.0903147
time: 2.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.4853067, 2.4473352, -2.1460061, 1.5129943, -4.9983010, 4.5933414
1: -2.3860233, 2.7283158, -1.5510125, 1.6763215, -4.0623446, 4.2793283
2: -3.4779963, 2.7229323, -2.0382948, 1.8424779, -5.3204741, 4.7612271
3: -4.1950302, 2.1269622, -2.4783916, 1.3712867, -5.5663171, 4.6053538
4: -4.4386516, 2.9296663, -2.5710292, 1.9505956, -6.3892469, 5.5006952
5: -3.5716043, 2.3697858, -2.2027714, 1.5578930, -5.1294975, 4.5725574
6: -4.2921257, 2.8144977, -2.4500713, 1.9257549, -6.2178807, 5.2645693
7: -3.2460740, 3.2257340, -2.0309029, 2.0493848, -5.2954588, 5.2566366
8: -4.7033777, 2.5684268, -2.7415049, 1.6671121, -6.3704901, 5.3099318
9: -3.1413107, 3.1758220, -1.9559568, 1.9950235, -5.1363344, 5.1317787

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949635, upper bound: 7.0898348
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949635, upper bound: 7.0898348
time: 2.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2113070, 2.9704983, -1.6535420, 1.2164484, -5.4277554, 4.6240406
1: -2.8210583, 3.2391477, -1.2471907, 1.3046579, -4.1257162, 4.4863386
2: -4.1861949, 3.1990931, -1.5204964, 1.5255494, -5.7117443, 4.7195892
3: -5.0715890, 2.5738735, -1.8410285, 1.1018996, -6.1734886, 4.4149017
4: -5.3326454, 3.3389144, -1.9428211, 1.6254830, -6.9581285, 5.2817354
5: -4.4420171, 2.7320628, -1.6999413, 1.2921932, -5.7342100, 4.4320040
6: -5.1272678, 3.0887017, -1.7902324, 1.7014731, -6.8287411, 4.8789339
7: -3.8680859, 3.8621020, -1.5880589, 1.6180925, -5.4861784, 5.4501610
8: -5.6511602, 2.9265049, -2.0408082, 1.3882965, -7.0394568, 4.9673128
9: -3.7477620, 3.7574701, -1.5267857, 1.5803237, -5.3280859, 5.2842560

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937861, upper bound: 7.0892591
time: 2.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937861, upper bound: 7.0892591
time: 3.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.4947109, 4.6415038, -1.6252153, 1.2002740, -7.6949849, 6.2667189
1: -4.2225828, 4.9810328, -1.2302003, 1.2832751, -5.5058579, 6.2112331
2: -6.5618954, 4.7261076, -1.4913107, 1.5072876, -8.0691833, 6.2174182
3: -7.8973041, 3.9265864, -1.8042994, 1.0864162, -8.9837208, 5.7308855
4: -8.3741846, 4.9187684, -1.9075618, 1.6071684, -9.9813528, 6.8263302
5: -6.9476790, 4.0531206, -1.6709419, 1.2775904, -8.2252693, 5.7240624
6: -8.1222029, 4.4661012, -1.7521672, 1.6901554, -9.8123579, 6.2182684
7: -5.8839903, 5.8597250, -1.5631514, 1.5935466, -7.4775372, 7.4228764
8: -8.8707256, 4.4392843, -2.0002806, 1.3734850, -10.2442102, 6.4395647
9: -5.7361798, 5.6981201, -1.5022744, 1.5566587, -7.2928386, 7.2003946

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0928197, upper bound: 7.0889357
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0928197, upper bound: 7.0889357
time: 3.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.6663671, 3.2989087, -2.0877483, 1.4736432, -6.1400104, 5.3866568
1: -3.1029539, 3.5872383, -1.5138625, 1.6318682, -4.7348223, 5.1011009
2: -4.6656966, 3.5021510, -1.9745770, 1.8047190, -6.4704156, 5.4767280
3: -5.6508298, 2.8377142, -2.4026415, 1.3392893, -6.9901190, 5.2403555
4: -5.9493184, 3.6501350, -2.4904389, 1.9099448, -7.8592634, 6.1405740
5: -4.9548154, 2.9897494, -2.1437464, 1.5240567, -6.4788723, 5.1334958
6: -5.7262878, 3.3430717, -2.3732150, 1.8918706, -7.6181583, 5.7162867
7: -4.2795696, 4.2699537, -1.9775679, 1.9986571, -6.2782269, 6.2475214
8: -6.3005600, 3.1984859, -2.6575549, 1.6310099, -7.9315701, 5.8560410
9: -4.1488614, 4.1531267, -1.9043918, 1.9449855, -6.0938468, 6.0575185

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943373, upper bound: 7.0892599
time: 2.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0943373, upper bound: 7.0892599
time: 2.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.9444790, 4.9653406, -2.0567026, 1.4538786, -8.3983574, 7.0220432
1: -4.5009375, 5.3248806, -1.4948196, 1.6082252, -6.1091628, 6.8197002
2: -7.0358934, 5.0259414, -1.9417101, 1.7844896, -8.8203831, 6.9676514
3: -8.4699783, 4.1875973, -2.3627014, 1.3219353, -9.7919140, 6.5502987
4: -8.9841976, 5.2262287, -2.4472713, 1.8889847, -10.8731823, 7.6735001
5: -7.4541774, 4.3077054, -2.1117840, 1.5067385, -8.9609156, 6.4194894
6: -8.7146912, 4.7144384, -2.3323886, 1.8771579, -10.5918493, 7.0468273
7: -6.2899466, 6.2632375, -1.9495122, 1.9712892, -8.2612362, 8.2127495
8: -9.5128212, 4.7070532, -2.6129375, 1.6139860, -11.1268072, 7.3199906
9: -6.1328669, 6.0894966, -1.8770078, 1.9179449, -8.0508118, 7.9665046

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0931771, upper bound: 7.0889357
time: 2.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0931771, upper bound: 7.0889357
time: 2.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.2110214, 2.2799141, -0.5297745, 0.5894055, -3.8004270, 2.8096886
1: -2.2106903, 2.5033202, -0.5504900, 0.5564717, -2.7671618, 3.0538101
2: -3.1589596, 2.5595167, -0.5467492, 0.7444847, -3.9034443, 3.1062658
3: -3.7989054, 1.9987085, -0.5155994, 0.4821867, -4.2810922, 2.5143080
4: -4.0021234, 2.7055497, -0.6304571, 0.8269848, -4.8291082, 3.3360069
5: -3.3283343, 2.1841297, -0.6495303, 0.6685611, -3.9968953, 2.8336601
6: -3.8826060, 2.6145062, -0.2492144, 1.3486676, -5.2312737, 2.8637207
7: -2.9794934, 2.9760151, -0.6846844, 0.6549921, -3.6344855, 3.6606994
8: -4.2735643, 2.3828790, -0.6571739, 0.7668773, -5.0404415, 3.0400529
9: -2.8885622, 2.9133992, -0.6107385, 0.6668303, -3.5553925, 3.5241377

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947621, upper bound: 7.0901832
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947621, upper bound: 7.0901832
time: 11.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.3390222, 3.7744038, -0.5183210, 0.5812716, -5.9202938, 4.2927246
1: -3.6826088, 4.0237012, -0.5415297, 0.5492020, -4.2318106, 4.5652308
2: -5.2676315, 3.9667764, -0.5379832, 0.7355621, -6.0031939, 4.5047598
3: -6.2960711, 3.2023740, -0.5064608, 0.4750292, -6.7711000, 3.7088346
4: -6.6399002, 4.1021347, -0.6184523, 0.8175472, -7.4574475, 4.7205868
5: -5.5385771, 3.4463162, -0.6398857, 0.6609597, -6.1995368, 4.0862017
6: -6.5193672, 3.9377618, -0.2358108, 1.3469011, -7.8662682, 4.1735725
7: -4.7790008, 4.7540808, -0.6766230, 0.6439427, -5.4229436, 5.4307036
8: -7.1148872, 3.7837644, -0.6456950, 0.7598924, -7.8747797, 4.4294596
9: -4.6297660, 4.6542473, -0.6016129, 0.6544952, -5.2842612, 5.2558603

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0928529, upper bound: 7.0896393
time: 3.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0928529, upper bound: 7.0896393
time: 2.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.6840832, 2.6094894, -0.9386261, 0.8337950, -4.5178781, 3.5481155
1: -2.4987230, 2.8519366, -0.8307748, 0.7952273, -3.2939503, 3.6827114
2: -3.6434407, 2.8664613, -0.8563201, 1.0619371, -4.7053776, 3.7227814
3: -4.3830996, 2.2645290, -0.9269466, 0.7205147, -5.1036143, 3.1914756
4: -4.6175966, 3.0159304, -1.0698823, 1.1357107, -5.7533073, 4.0858126
5: -3.8453131, 2.4545114, -1.0002662, 0.9193050, -4.7646179, 3.4547777
6: -4.4836683, 2.8696830, -0.7836188, 1.4361819, -5.9198503, 3.6533017
7: -3.3960724, 3.3850298, -0.9874184, 1.0277476, -4.4238200, 4.3724480
8: -4.9264698, 2.6596110, -1.1201810, 1.0017483, -5.9282179, 3.7797918
9: -3.2909813, 3.3102939, -0.9305614, 1.0250351, -4.3160162, 4.2408552

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0950684, upper bound: 7.0901832
time: 3.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0950684, upper bound: 7.0901832
time: 3.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.9342494, 4.1835732, -0.9145581, 0.8209035, -6.7551527, 5.0981312
1: -4.1182556, 4.4450226, -0.8161835, 0.7825176, -4.9007730, 5.2612062
2: -5.8617492, 4.3558989, -0.8379371, 1.0449359, -6.9066849, 5.1938362
3: -7.0088553, 3.5300035, -0.8980969, 0.7087190, -7.7175741, 4.4281006
4: -7.3821425, 4.4861541, -1.0437737, 1.1199398, -8.5020828, 5.5299277
5: -6.1626015, 3.7947378, -0.9788906, 0.9076622, -7.0702639, 4.7736282
6: -7.2551150, 4.2499781, -0.7546921, 1.4318485, -8.6869640, 5.0046701
7: -5.2877302, 5.2648735, -0.9695115, 1.0087241, -6.2964544, 6.2343850
8: -7.9043279, 4.1335773, -1.0941569, 0.9904779, -8.8948059, 5.2277341
9: -5.1190510, 5.1514482, -0.9116818, 1.0075088, -6.1265597, 6.0631299

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0930090, upper bound: 7.0896415
time: 2.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0930090, upper bound: 7.0896415
time: 2.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5176697, 4.5529728, -0.4961825, 0.5664194, -7.0840893, 5.0491552
1: -4.5538263, 4.8469238, -0.5228843, 0.5354913, -5.0893178, 5.3698082
2: -6.4458728, 4.7109137, -0.5187782, 0.7194453, -7.1653180, 5.2296920
3: -7.7646341, 3.8283896, -0.4894353, 0.4631696, -8.2278042, 4.3178248
4: -8.1290512, 4.8237362, -0.5945351, 0.7974223, -8.9264736, 5.4182711
5: -6.8027792, 4.1124072, -0.6213492, 0.6452179, -7.4479971, 4.7337565
6: -7.9454260, 4.4206924, -0.2065223, 1.3388991, -9.2843246, 4.6272149
7: -5.7947621, 5.7942543, -0.6595368, 0.6236882, -6.4184504, 6.4537911
8: -8.6717978, 4.3414593, -0.6232234, 0.7407423, -9.4125404, 4.9646826
9: -5.5970511, 5.6570559, -0.5847055, 0.6319005, -6.2289515, 6.2417612

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907834, upper bound: 7.0888620
time: 2.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907834, upper bound: 7.0888620
time: 2.26 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9391747, 6.2385244, -0.4876108, 0.5595744, -9.4987488, 6.7261353
1: -6.3003216, 6.5580392, -0.5153219, 0.5300806, -6.8304024, 7.0733609
2: -8.8384676, 6.3118196, -0.5117576, 0.7123882, -9.5508556, 6.8235769
3: -10.6538820, 5.1874638, -0.4833500, 0.4579881, -11.1118698, 5.6708136
4: -11.1177816, 6.4080477, -0.5846245, 0.7903515, -11.9081335, 6.9926720
5: -9.3008623, 5.5586820, -0.6134032, 0.6390173, -9.9398794, 6.1720853
6: -10.9300547, 5.8626952, -0.1965175, 1.3377221, -12.2677765, 6.0592127
7: -7.8380108, 7.8331041, -0.6530887, 0.6145822, -8.4525928, 8.4861927
8: -11.8826466, 5.9059491, -0.6140817, 0.7353472, -12.6179934, 6.5200310
9: -7.5659108, 7.6643534, -0.5777442, 0.6215581, -8.1874685, 8.2420979

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0884682, upper bound: 7.0885707
time: 2.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0884682, upper bound: 7.0885707
time: 10.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.0104852, 4.8929129, -0.8611449, 0.7940040, -7.8044891, 5.7540579
1: -4.9153929, 5.1972170, -0.7826685, 0.7547370, -5.6701298, 5.9798856
2: -6.9396634, 5.0337429, -0.7967097, 1.0096085, -7.9492722, 5.8304524
3: -8.3620481, 4.0978723, -0.8364654, 0.6834968, -9.0455446, 4.9343376
4: -8.7479334, 5.1433816, -0.9884108, 1.0830436, -9.8309765, 6.1317925
5: -7.3221941, 4.4029522, -0.9339492, 0.8811681, -8.2033625, 5.3369017
6: -8.5573139, 4.6952868, -0.6898898, 1.4177061, -9.9750204, 5.3851767
7: -6.2185612, 6.2185888, -0.9298481, 0.9683119, -7.1868730, 7.1484370
8: -9.3298721, 4.6310310, -1.0382413, 0.9592140, -10.2890863, 5.6692724
9: -6.0024519, 6.0723181, -0.8713965, 0.9705028, -6.9729548, 6.9437146

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
time: 2.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
time: 2.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.4507828, 6.5901933, -0.8408775, 0.7836053, -10.2343884, 7.4310708
1: -6.6752048, 6.9204264, -0.7701228, 0.7449175, -7.4201221, 7.6905494
2: -9.3498802, 6.6461449, -0.7824740, 0.9955488, -10.3454294, 7.4286189
3: -11.2743912, 5.4676161, -0.8142562, 0.6732865, -11.9476776, 6.2818723
4: -11.7584543, 6.7394047, -0.9675936, 1.0701194, -12.8285732, 7.7069983
5: -9.8391800, 5.8594971, -0.9172844, 0.8709570, -10.7101374, 6.7767816
6: -11.5642586, 6.1452150, -0.6670545, 1.4145955, -12.9788542, 6.8122697
7: -8.2782106, 8.2743187, -0.9149129, 0.9525325, -9.2307434, 9.1892319
8: -12.5627823, 6.2062726, -1.0174141, 0.9499651, -13.5127478, 7.2236867
9: -7.9868555, 8.0948753, -0.8564568, 0.9555174, -8.9423733, 8.9513321

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885593, upper bound: 7.0885707
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885593, upper bound: 7.0885707
time: 2.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.4697886, 2.4665432, -1.7698430, 1.2826169, -4.7524052, 4.2363863
1: -2.3717492, 2.6990423, -1.3180175, 1.3914299, -3.7631791, 4.0170598
2: -3.4286721, 2.7324851, -1.6411316, 1.5999446, -5.0286169, 4.3736167
3: -4.1236510, 2.1481628, -1.9922434, 1.1642427, -5.2878938, 4.1404061
4: -4.3452940, 2.8799694, -2.0869639, 1.7007020, -6.0459957, 4.9669333
5: -3.6188331, 2.3319838, -1.8172286, 1.3527085, -4.9715414, 4.1492124
6: -4.2212982, 2.7632983, -1.9456466, 1.7516837, -5.9729819, 4.7089448
7: -3.2112536, 3.2039514, -1.6908622, 1.7164122, -4.9276657, 4.8948135
8: -4.6416903, 2.5447941, -2.2057178, 1.4527218, -6.0944118, 4.7505121
9: -3.1131959, 3.1359715, -1.6269767, 1.6767399, -4.7899361, 4.7629480

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951315, upper bound: 7.0901832
time: 2.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0951315, upper bound: 7.0901832
time: 2.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.6222391, 3.9705167, -1.7376676, 1.2641417, -6.8863807, 5.7081842
1: -3.8902407, 4.2253852, -1.2985772, 1.3671811, -5.2574215, 5.5239625
2: -5.5514612, 4.1530108, -1.6080146, 1.5792084, -7.1306696, 5.7610254
3: -6.6359577, 3.3573787, -1.9505713, 1.1466892, -7.7826471, 5.3079500
4: -6.9942436, 4.2857966, -2.0468636, 1.6798346, -8.6740780, 6.3326602
5: -5.8378320, 3.6133826, -1.7842983, 1.3359853, -7.1738172, 5.3976808
6: -6.8721662, 4.0908628, -1.9022748, 1.7383146, -8.6104813, 5.9931374
7: -5.0211954, 4.9975114, -1.6626431, 1.6885529, -6.7097483, 6.6601543
8: -7.4954553, 3.9560661, -2.1597254, 1.4356561, -8.9311113, 6.1157913
9: -4.8625231, 4.8929787, -1.5992357, 1.6497662, -6.5122890, 6.4922142

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0932615, upper bound: 7.0896393
time: 2.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0932615, upper bound: 7.0896393
time: 2.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9667621, 2.8073359, -2.2116721, 1.5554943, -5.5222564, 5.0190077
1: -2.7062144, 3.0546880, -1.5914166, 1.7263584, -4.4325728, 4.6461048
2: -3.9253421, 3.0530560, -2.1080325, 1.8855445, -5.8108864, 5.1610885
3: -4.7226396, 2.4194071, -2.5627816, 1.4086154, -6.1312551, 4.9821887
4: -4.9718752, 3.1972265, -2.6617022, 1.9945989, -6.9664741, 5.8589287
5: -4.1449556, 2.6218650, -2.2708526, 1.5947237, -5.7396793, 4.8927174
6: -4.8357668, 3.0278730, -2.5374100, 1.9562497, -6.7920165, 5.5652828
7: -3.6371267, 3.6287351, -2.0907955, 2.1074991, -5.7446260, 5.7195306
8: -5.3069520, 2.8348813, -2.8365843, 1.7027395, -7.0096912, 5.6714659
9: -3.5241864, 3.5470982, -2.0141969, 2.0531116, -5.5772982, 5.5612950

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953006, upper bound: 7.0901832
time: 3.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953006, upper bound: 7.0901832
time: 3.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2114468, 4.3770704, -2.1752210, 1.5315870, -7.7430339, 6.5522914
1: -4.3226118, 4.6418419, -1.5689381, 1.6985483, -6.0211601, 6.2107801
2: -6.1389832, 4.5381784, -2.0693474, 1.8616605, -8.0006437, 6.6075258
3: -7.3708010, 3.6818130, -2.5158074, 1.3880246, -8.7588253, 6.1976204
4: -7.7281275, 4.6658616, -2.6112344, 1.9698513, -9.6979790, 7.2770958
5: -6.4549098, 3.9623995, -2.2331553, 1.5740515, -8.0289612, 6.1955547
6: -7.6000004, 4.4014740, -2.4889996, 1.9382238, -9.5382242, 6.8904734
7: -5.5243397, 5.5083823, -2.0576048, 2.0753345, -7.5996742, 7.5659871
8: -8.2755108, 4.3056574, -2.7839804, 1.6824312, -9.9579420, 7.0896378
9: -5.3467064, 5.3928757, -1.9819450, 2.0209832, -7.3676896, 7.3748207

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0933477, upper bound: 7.0896415
time: 2.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0933477, upper bound: 7.0896415
time: 2.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.7933245, 4.7442265, -1.6786215, 1.2311008, -8.0244255, 6.4228477
1: -4.7558031, 5.0431194, -1.2622060, 1.3236530, -6.0794563, 6.3053255
2: -6.7217417, 4.8923140, -1.5465782, 1.5419636, -8.2637053, 6.4388924
3: -8.0968122, 3.9793382, -1.8737255, 1.1158428, -9.2126551, 5.8530636
4: -8.4737997, 5.0033960, -1.9742053, 1.6411315, -10.1149311, 6.9776011
5: -7.0935369, 4.2756882, -1.7262121, 1.3048799, -8.3984165, 6.0019002
6: -8.2890968, 4.5773735, -1.8241998, 1.7102758, -9.9993725, 6.4015732
7: -6.0306525, 6.0306726, -1.6101836, 1.6400472, -7.6706996, 7.6408563
8: -9.0421619, 4.5105114, -2.0774312, 1.4006890, -10.4428511, 6.5879426
9: -5.8231983, 5.8895640, -1.5487986, 1.6017630, -7.4249611, 7.4383626

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907856, upper bound: 7.0888619
time: 2.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907856, upper bound: 7.0888619
time: 2.20 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.2342987, 6.4419250, -1.6499796, 1.2147548, -10.4490538, 8.0919046
1: -6.5167108, 6.7670178, -1.2450287, 1.3020453, -7.8187561, 8.0120468
2: -9.1330395, 6.5050874, -1.5170803, 1.5235074, -10.6565466, 8.0221672
3: -11.0104961, 5.3491459, -1.8366029, 1.1001964, -12.1106930, 7.1857491
4: -11.4859829, 6.5996380, -1.9385650, 1.6226072, -13.1085901, 8.5382032
5: -9.6120224, 5.7324696, -1.6968964, 1.2901167, -10.9021387, 7.4293661
6: -11.2970982, 6.0266757, -1.7857138, 1.6987703, -12.9958687, 7.8123894
7: -8.0910816, 8.0875759, -1.5850059, 1.6152408, -9.7063227, 9.6725817
8: -12.2764454, 6.0839972, -2.0364685, 1.3857191, -13.6621647, 8.1204662
9: -7.8084302, 7.9129834, -1.5240264, 1.5778446, -9.3862743, 9.4370098

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885028, upper bound: 7.0885707
time: 2.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885028, upper bound: 7.0885707
time: 2.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.2961335, 5.0908861, -2.1167560, 1.4923233, -8.7884569, 7.2076421
1: -5.1246772, 5.4004598, -1.5316470, 1.6539859, -6.7786632, 6.9321070
2: -7.2253714, 5.2216301, -2.0054765, 1.8237791, -9.0491505, 7.2271066
3: -8.7063370, 4.2542667, -2.4398654, 1.3556994, -10.0620365, 6.6941319
4: -9.1052446, 5.3292360, -2.5303514, 1.9290528, -11.0342979, 7.8595877
5: -7.6236682, 4.5718946, -2.1740985, 1.5399388, -9.1636066, 6.7459931
6: -8.9132900, 4.8567352, -2.4116676, 1.9044603, -10.8177500, 7.2684031
7: -6.4630380, 6.4637318, -2.0036910, 2.0245159, -8.4875536, 8.4674225
8: -9.7134371, 4.8056231, -2.6997139, 1.6461960, -11.3596334, 7.5053368
9: -6.2367306, 6.3133192, -1.9302030, 1.9704947, -8.2072258, 8.2435226

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888619
time: 2.22 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
time: 2.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7344666, 6.7860541, -2.0843773, 1.4716997, -11.2061663, 8.8704319
1: -6.8832555, 7.1218166, -1.5117989, 1.6293625, -8.5126181, 8.6336155
2: -9.6335344, 6.8322787, -1.9712862, 1.8026500, -11.4361839, 8.8035650
3: -11.6176605, 5.6229692, -2.3982997, 1.3376570, -12.9553175, 8.0212688
4: -12.1131811, 6.9236274, -2.4854815, 1.9071462, -14.0203276, 9.4091091
5: -10.1388721, 6.0268569, -2.1408165, 1.5218838, -11.6607561, 8.1676731
6: -11.9174957, 6.3030415, -2.3691020, 1.8890892, -13.8065853, 8.6721439
7: -8.5218477, 8.5189600, -1.9745618, 1.9960549, -10.5179024, 10.4935217
8: -12.9425659, 6.3773289, -2.6533375, 1.6283700, -14.5709362, 9.0306664
9: -8.2200489, 8.3343592, -1.9017084, 1.9423119, -10.1623611, 10.2360678

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885707, upper bound: 7.0885707
time: 2.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0885707, upper bound: 7.0885707
time: 3.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.41 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0959462, upper bound: 7.0903147
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0959462, upper bound: 7.0903147
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0947559, upper bound: 7.0898348
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0947559, upper bound: 7.0898348
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0937021, upper bound: 7.0892591
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0937021, upper bound: 7.0892591
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0943064, upper bound: 7.0892599
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0943064, upper bound: 7.0892599
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0931676, upper bound: 7.0889357
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0931676, upper bound: 7.0889357
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0958112, upper bound: 7.0903147
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0958112, upper bound: 7.0903147
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0947962, upper bound: 7.0898348
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0947962, upper bound: 7.0898348
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0960948, upper bound: 7.0903147
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0960948, upper bound: 7.0903147
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0949635, upper bound: 7.0898348
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0949635, upper bound: 7.0898348
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0937861, upper bound: 7.0892591
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0937861, upper bound: 7.0892591
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0928197, upper bound: 7.0889357
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0928197, upper bound: 7.0889357
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0943373, upper bound: 7.0892599
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0943373, upper bound: 7.0892599
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0931771, upper bound: 7.0889357
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0931771, upper bound: 7.0889357
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0947621, upper bound: 7.0901832
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0947621, upper bound: 7.0901832
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0928529, upper bound: 7.0896393
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0928529, upper bound: 7.0896393
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0950684, upper bound: 7.0901832
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0950684, upper bound: 7.0901832
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0930090, upper bound: 7.0896415
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0930090, upper bound: 7.0896415
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0907834, upper bound: 7.0888620
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0907834, upper bound: 7.0888620
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0884682, upper bound: 7.0885707
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0884682, upper bound: 7.0885707
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0885593, upper bound: 7.0885707
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0885593, upper bound: 7.0885707
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0951315, upper bound: 7.0901832
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0951315, upper bound: 7.0901832
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0932615, upper bound: 7.0896393
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0932615, upper bound: 7.0896393
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0953006, upper bound: 7.0901832
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0953006, upper bound: 7.0901832
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0933477, upper bound: 7.0896415
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0933477, upper bound: 7.0896415
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0907856, upper bound: 7.0888619
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0907856, upper bound: 7.0888619
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0885028, upper bound: 7.0885707
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0885028, upper bound: 7.0885707
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888619
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0910820, upper bound: 7.0888620
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0885707, upper bound: 7.0885707
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.41
Output dim: 6, lower bound: -7.0885707, upper bound: 7.0885707

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.9007107, 0.8166325, -0.4090952, 0.4985660, -1.3992767, 1.2257278
1: -0.8046172, 0.7802006, -0.4487007, 0.4831146, -1.2877318, 1.2289013
2: -0.8202603, 1.0439324, -0.4441292, 0.6450853, -1.4653456, 1.4880617
3: -0.8737449, 0.7098458, -0.4266779, 0.4058384, -1.2795832, 1.1365237
4: -1.0246694, 1.1293843, -0.4933744, 0.7149383, -1.7396077, 1.6227586
5: -0.9702371, 0.9062064, -0.5398233, 0.5766564, -1.5468936, 1.4460297
6: -0.7667005, 1.4258229, -0.0874102, 1.3188030, -2.0855036, 1.5132329
7: -0.9592206, 1.0084324, -0.5921195, 0.5317180, -1.4909387, 1.6005518
8: -1.0847286, 0.9787750, -0.5391330, 0.6588457, -1.7435744, 1.5179079
9: -0.9018761, 1.0038450, -0.5107148, 0.5286460, -1.4305221, 1.5145597

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 4.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.9007107, 0.8166325, -1.3237468, 1.1393644, -2.0400751, 2.1403794
1: -0.8046172, 0.7802006, -1.1788528, 1.0629904, -1.8676076, 1.9590534
2: -0.8202603, 1.0439324, -1.1900574, 1.3235328, -2.1437931, 2.2339897
3: -0.8737449, 0.7098458, -1.1477797, 0.9203749, -1.7941198, 1.8576255
4: -1.0246694, 1.1293843, -1.4430188, 1.5579064, -2.5825758, 2.5724030
5: -0.9702371, 0.9062064, -1.3169563, 1.2077395, -2.1779766, 2.2231627
6: -0.7667005, 1.4258229, -1.2602373, 1.5797923, -2.3464928, 2.6860602
7: -0.9592206, 1.0084324, -1.2854741, 1.3790600, -2.3382807, 2.2939065
8: -1.0847286, 0.9787750, -1.4268832, 1.4047835, -2.4895120, 2.4056582
9: -0.9018761, 1.0038450, -1.2211468, 1.5062723, -2.4081483, 2.2249918

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 3.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 3.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -2.7110963, 1.7823443, -0.4033169, 0.4930175, -3.2041137, 2.1856613
1: -1.8644289, 2.0951281, -0.4445010, 0.4787333, -2.3431621, 2.5396290
2: -2.5329409, 2.1978054, -0.4397661, 0.6391019, -3.1720428, 2.6375716
3: -3.1576500, 1.6583755, -0.4226133, 0.4016505, -3.5593004, 2.0809889
4: -3.1931124, 2.3768959, -0.4871579, 0.7082736, -3.9013860, 2.8640537
5: -2.7069514, 1.8717282, -0.5344424, 0.5710384, -3.2779899, 2.4061706
6: -3.2545156, 2.2633750, -0.0800045, 1.3182049, -4.5727205, 2.3433795
7: -2.5033851, 2.4735458, -0.5868829, 0.5269530, -3.0303380, 3.0604286
8: -3.4588385, 2.0934308, -0.5347995, 0.6526442, -4.1114826, 2.6282301
9: -2.4083297, 2.4128132, -0.5046408, 0.5229746, -2.9313042, 2.9174540

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 2.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 2.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -2.7110963, 1.7823443, -1.2398620, 1.1142282, -3.8253245, 3.0222063
1: -1.8644289, 2.0951281, -1.1677530, 1.0157888, -2.8802176, 3.2628810
2: -2.5329409, 2.1978054, -1.1602858, 1.3103843, -3.8433251, 3.3580914
3: -3.1576500, 1.6583755, -1.0506182, 0.9029561, -4.0606060, 2.7089937
4: -3.1931124, 2.3768959, -1.4284625, 1.5127230, -4.7058353, 3.8053584
5: -2.7069514, 1.8717282, -1.2990724, 1.1930728, -3.9000242, 3.1708007
6: -3.2545156, 2.2633750, -1.2331083, 1.5734069, -4.8279228, 3.4964833
7: -2.5033851, 2.4735458, -1.2532629, 1.3666918, -3.8700769, 3.7268085
8: -3.4588385, 2.0934308, -1.4140700, 1.3779093, -4.8367476, 3.5075006
9: -2.4083297, 2.4128132, -1.1839966, 1.4636346, -3.8719645, 3.5968099

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 3.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 3.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.2656287, 1.0083714, -0.7008098, 0.7044352, -1.9700639, 1.7091812
1: -1.0195036, 1.0204811, -0.6806698, 0.6673300, -1.6868336, 1.7011509
2: -1.1430192, 1.2835000, -0.6771708, 0.8876522, -2.0306714, 1.9606707
3: -1.3459476, 0.8986995, -0.6694626, 0.5964780, -1.9424256, 1.5681621
4: -1.4721084, 1.3677002, -0.8159027, 0.9748777, -2.4469862, 2.1836028
5: -1.3211970, 1.0928774, -0.7967024, 0.7874779, -2.1086750, 1.8895798
6: -1.2669408, 1.5314304, -0.4875100, 1.3836827, -2.6506236, 2.0189404
7: -1.2533193, 1.3006904, -0.8103018, 0.8287641, -2.0820835, 2.1109922
8: -1.5347754, 1.1699867, -0.8564823, 0.8693106, -2.4040859, 2.0264690
9: -1.2029952, 1.2798890, -0.7478592, 0.8389358, -2.0419309, 2.0277481

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903106
time: 2.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 3.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.2656287, 1.0083714, -2.5169356, 1.6614537, -2.9270825, 3.5253069
1: -1.0195036, 1.0204811, -1.8082145, 1.6348959, -2.6543994, 2.8286958
2: -1.1430192, 1.2835000, -2.0839968, 2.1651585, -3.3081777, 3.3674967
3: -1.3459476, 0.8986995, -2.7978764, 1.4922740, -2.8382215, 3.6965759
4: -1.4721084, 1.3677002, -2.7679570, 2.2494695, -3.7215779, 4.1356573
5: -1.3211970, 1.0928774, -2.3944030, 1.6980020, -3.0191989, 3.4872804
6: -1.2669408, 1.5314304, -2.7990150, 1.8430698, -3.1100106, 4.3304453
7: -1.2533193, 1.3006904, -2.1545815, 2.2722514, -3.5255706, 3.4552720
8: -1.5347754, 1.1699867, -2.8277743, 1.8852696, -3.4200449, 3.9977610
9: -1.2029952, 1.2798890, -2.1712575, 2.1470280, -3.3500233, 3.4511466

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903106
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0955115, upper bound: 7.0903147
time: 3.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.2414737, 2.1565280, -0.6849285, 0.6940400, -3.9355137, 2.8414564
1: -2.2303569, 2.5118790, -0.6695794, 0.6572542, -2.8876112, 3.1814585
2: -3.1797576, 2.5396690, -0.6643737, 0.8738232, -4.0535808, 3.2040427
3: -3.8472333, 1.9607602, -0.6528382, 0.5864832, -4.4337163, 2.6135983
4: -3.8699775, 2.7453194, -0.7975702, 0.9636544, -4.8336320, 3.5428896
5: -3.2844045, 2.1705277, -0.7823356, 0.7764717, -4.0608764, 2.9528632
6: -3.9639344, 2.5611429, -0.4676026, 1.3812069, -5.3451414, 3.0287454
7: -3.0012994, 2.9981453, -0.7982452, 0.8126567, -3.8139560, 3.7963905
8: -4.2727776, 2.4142337, -0.8363209, 0.8609849, -5.1337624, 3.2505546
9: -2.9014208, 2.8849616, -0.7348598, 0.8232704, -3.7246914, 3.6198213

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0897964
time: 3.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.2414737, 2.1565280, -2.4817572, 1.6453264, -4.8867998, 4.6382852
1: -2.2303569, 2.5118790, -1.7830020, 1.6082875, -3.8386445, 4.2948809
2: -3.1797576, 2.5396690, -2.0320821, 2.1163285, -5.2960863, 4.5717511
3: -3.8472333, 1.9607602, -2.6494355, 1.4778407, -5.3250742, 4.6101956
4: -3.8699775, 2.7453194, -2.6851282, 2.2307794, -6.1007566, 5.4304476
5: -3.2844045, 2.1705277, -2.3210683, 1.6838981, -4.9683027, 4.4915962
6: -3.9639344, 2.5611429, -2.6697395, 1.8367981, -5.8007326, 5.2308826
7: -3.0012994, 2.9981453, -2.1337690, 2.2365851, -5.2378845, 5.1319141
8: -4.2727776, 2.4142337, -2.7329912, 1.8676267, -6.1404042, 5.1472249
9: -2.9014208, 2.8849616, -2.1015148, 2.1192646, -5.0206852, 4.9864764

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0897964
time: 3.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944760, upper bound: 7.0898348
time: 3.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9496837, 2.7732000, -0.3902727, 0.4824503, -4.4321342, 3.1634727
1: -2.6545053, 3.0374718, -0.4347854, 0.4691958, -3.1237011, 3.4722571
2: -3.9045963, 3.0238886, -0.4299823, 0.6269802, -4.5315766, 3.4538708
3: -4.7330346, 2.4226484, -0.4136333, 0.3929270, -5.1259618, 2.8362818
4: -4.9728127, 3.1564448, -0.4746168, 0.6921701, -5.6649828, 3.6310616
5: -4.1421041, 2.5822048, -0.5226542, 0.5588524, -4.7009563, 3.1048589
6: -4.7790027, 2.9381104, -0.0588450, 1.3129810, -6.0919838, 2.9969554
7: -3.6299500, 3.6251717, -0.5752267, 0.5163258, -4.1462760, 4.2003984
8: -5.2738137, 2.7618079, -0.5244602, 0.6350237, -5.9088373, 3.2862682
9: -3.5135031, 3.5259776, -0.4925672, 0.5114211, -4.0249243, 4.0185447

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937020, upper bound: 7.0891974
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937020, upper bound: 7.0892591
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9496837, 2.7732000, -1.1989880, 1.0943317, -5.0440154, 3.9721880
1: -2.6545053, 3.0374718, -1.1482354, 0.9853776, -3.6398828, 4.1857071
2: -3.9045963, 3.0238886, -1.1093506, 1.2930180, -5.1976142, 4.1332393
3: -4.7330346, 2.4226484, -0.9861420, 0.8822111, -5.6152458, 3.4087906
4: -4.9728127, 3.1564448, -1.3871480, 1.4622853, -6.4350977, 4.5435929
5: -4.1421041, 2.5822048, -1.2779224, 1.1664710, -5.3085752, 3.8601272
6: -4.7790027, 2.9381104, -1.1959035, 1.5511272, -6.3301296, 4.1340141
7: -3.6299500, 3.6251717, -1.2075286, 1.3450036, -4.9749537, 4.8327003
8: -5.2738137, 2.7618079, -1.3776381, 1.3454093, -6.6192231, 4.1394463
9: -3.5135031, 3.5259776, -1.1632454, 1.4227419, -4.9362450, 4.6892233

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937020, upper bound: 7.0891974
time: 3.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937020, upper bound: 7.0892591
time: 2.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.2343907, 4.4305501, -0.3863343, 0.4777587, -6.7121496, 4.8168845
1: -4.0474377, 4.7808418, -0.4312763, 0.4661785, -4.5136161, 5.2121181
2: -6.2693372, 4.5503922, -0.4263760, 0.6225609, -6.8918982, 4.9767680
3: -7.5463839, 3.7759576, -0.4102235, 0.3893918, -7.9357758, 4.1861811
4: -8.0034208, 4.7335567, -0.4706498, 0.6866013, -8.6900225, 5.2042065
5: -6.6393270, 3.9045503, -0.5181215, 0.5542920, -7.1936193, 4.4226718
6: -7.7710896, 4.3196278, -0.0534951, 1.3127621, -9.0838518, 4.3731227
7: -5.6423306, 5.6192970, -0.5707908, 0.5127621, -6.1550927, 6.1900878
8: -8.4927101, 4.2740993, -0.5208266, 0.6308043, -9.1235142, 4.7949257
9: -5.4949799, 5.4606066, -0.4886872, 0.5065470, -6.0015268, 5.9492936

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0924929, upper bound: 7.0884711
time: 2.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
time: 3.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.2343907, 4.4305501, -1.1909876, 1.0450057, -7.2793965, 5.6215377
1: -4.0474377, 4.7808418, -1.1408396, 0.8914202, -4.9388580, 5.9216814
2: -6.2693372, 4.5503922, -1.1015000, 1.2864854, -7.5558224, 5.6518922
3: -7.5463839, 3.7759576, -0.9804832, 0.8766590, -8.4230433, 4.7564406
4: -8.0034208, 4.7335567, -1.3778057, 1.4543033, -9.4577236, 6.1113625
5: -6.6393270, 3.9045503, -1.2503517, 1.1607733, -7.8001003, 5.1549020
6: -7.7710896, 4.3196278, -1.1557157, 1.5499505, -9.3210402, 5.4753437
7: -5.6423306, 5.6192970, -1.1606320, 1.3365208, -6.9788513, 6.7799292
8: -8.4927101, 4.2740993, -1.3314416, 1.3406148, -9.8333244, 5.6055412
9: -5.4949799, 5.4606066, -1.1153810, 1.4130840, -6.9080639, 6.5759878

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0924929, upper bound: 7.0884711
time: 2.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0927921, upper bound: 7.0889357
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.4011250, 3.0958078, -0.6495361, 0.6723070, -5.0734320, 3.7453439
1: -2.9321208, 3.3827810, -0.6445714, 0.6344956, -3.5666163, 4.0273523
2: -4.3777075, 3.3242993, -0.6343850, 0.8460617, -5.2237692, 3.9586844
3: -5.3047566, 2.6843410, -0.6152817, 0.5658028, -5.8705597, 3.2996225
4: -5.5820398, 3.4644499, -0.7570789, 0.9364357, -6.5184755, 4.2215290
5: -4.6485653, 2.8379536, -0.7518046, 0.7521848, -5.4007502, 3.5897582
6: -5.3704548, 3.1902380, -0.4204930, 1.3709595, -6.7414141, 3.6107311
7: -4.0371370, 4.0287180, -0.7702578, 0.7808002, -4.8179374, 4.7989759
8: -5.9174786, 3.0312867, -0.7926763, 0.8363767, -6.7538552, 3.8239629
9: -3.9097035, 3.9169817, -0.7065282, 0.7904750, -4.7001786, 4.6235099

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942738, upper bound: 7.0891974
time: 5.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942738, upper bound: 7.0892599
time: 3.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.4011250, 3.0958078, -2.3454857, 1.5431017, -5.9442267, 5.4412937
1: -2.9321208, 3.3827810, -1.7144529, 1.4912921, -4.4234128, 5.0972338
2: -4.3777075, 3.3242993, -1.8701184, 2.0362079, -6.4139156, 5.1944180
3: -5.3047566, 2.6843410, -2.4172652, 1.4393550, -6.7441115, 5.1016064
4: -5.5820398, 3.4644499, -2.4704590, 2.1310039, -7.7130437, 5.9349089
5: -4.6485653, 2.8379536, -2.1428828, 1.6522942, -6.3008595, 4.9808364
6: -5.3704548, 3.1902380, -2.5133042, 1.7721031, -7.1425581, 5.7035422
7: -4.0371370, 4.0287180, -2.0445740, 2.1246879, -6.1618252, 6.0732918
8: -5.9174786, 3.0312867, -2.5774388, 1.7940478, -7.7115264, 5.6087255
9: -3.9097035, 3.9169817, -1.9728358, 2.0566916, -5.9663954, 5.8898172

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942738, upper bound: 7.0891974
time: 6.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0942738, upper bound: 7.0892599
time: 3.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.6910205, 4.7560925, -0.6362916, 0.6631182, -7.3541389, 5.3923841
1: -4.3279586, 5.1300368, -0.6348696, 0.6256347, -4.9535933, 5.7649064
2: -6.7480650, 4.8546081, -0.6232367, 0.8348855, -7.5829506, 5.4778447
3: -8.1249905, 4.0409760, -0.6019288, 0.5571119, -8.6821022, 4.6429048
4: -8.6203232, 5.0448637, -0.7415620, 0.9267427, -9.5470657, 5.7864256
5: -7.1515841, 4.1631036, -0.7400854, 0.7427610, -7.8943453, 4.9031892
6: -8.3719692, 4.5666485, -0.4040335, 1.3691851, -9.7411547, 4.9706821
7: -6.0535264, 6.0281744, -0.7598499, 0.7677536, -6.8212800, 6.7880244
8: -9.1442451, 4.5454135, -0.7759017, 0.8295015, -9.9737463, 5.3213153
9: -5.8960967, 5.8566270, -0.6960937, 0.7768081, -6.6729050, 6.5527205

Time for backsubstitution: 1.21 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=7.834464073181152
rel_dist={6: [-7.107152351905672, 7.10715235190567]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054165, upper bound: 7.1052182
time: 7.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1051752, upper bound: 7.1051752
time: 4.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.16
Output dim: 6, lower bound: -7.1054165, upper bound: 7.1052182
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.16
Output dim: 6, lower bound: -7.1051752, upper bound: 7.1051752

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.3117657, 2.3313293, -3.4660492, 2.4382138, -5.7499795, 5.7973785
1: -2.2746105, 2.5741804, -2.3672941, 2.6868043, -4.9614148, 4.9414744
2: -3.2746129, 2.6151695, -3.4315472, 2.7132530, -5.9878659, 6.0467167
3: -3.9685380, 2.0463929, -4.1567225, 2.1315551, -6.1000929, 6.2031155
4: -4.1542244, 2.7427235, -4.3512282, 2.8449337, -6.9991579, 7.0939517
5: -3.4665380, 2.2262321, -3.6319153, 2.3154886, -5.7820263, 5.8581476
6: -3.9971840, 2.5632598, -4.1924872, 2.6465175, -6.6437016, 6.7557468
7: -3.0885596, 3.0902281, -3.2221687, 3.2219276, -6.3104873, 6.3123970
8: -4.4174242, 2.3710141, -4.6264791, 2.4640045, -6.8814287, 6.9974933
9: -2.9865417, 3.0139289, -3.1165490, 3.1414828, -6.1280246, 6.1304779

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054165, upper bound: 7.1052182
time: 3.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054165, upper bound: 7.1052182
time: 4.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.0443048, 4.2405229, -3.5538156, 2.4968715, -8.5411758, 7.7943382
1: -4.2253499, 4.5275450, -2.4185190, 2.7499542, -6.9753041, 6.9460640
2: -5.9989243, 4.4117823, -3.5193048, 2.7684779, -8.7674026, 7.9310870
3: -7.2230110, 3.5636163, -4.2614303, 2.1800687, -9.4030800, 7.8250465
4: -7.5507116, 4.5344086, -4.4605923, 2.9015081, -10.4522200, 8.9950008
5: -6.3241363, 3.8430071, -3.7254462, 2.3654728, -8.6896095, 7.5684533
6: -7.4014902, 4.1482449, -4.3019538, 2.6883779, -10.0898685, 8.4501991
7: -5.4129872, 5.4127798, -3.2969961, 3.2964911, -8.7094784, 8.7097759
8: -8.0652170, 4.0733213, -4.7438707, 2.5133395, -10.5785561, 8.8171921
9: -5.2316179, 5.2847004, -3.1895313, 3.2133219, -8.4449396, 8.4742317

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1051752, upper bound: 7.1051752
time: 3.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1051752, upper bound: 7.1051752
time: 3.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.86
Output dim: 6, lower bound: -7.1054165, upper bound: 7.1052182
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.86
Output dim: 6, lower bound: -7.1054165, upper bound: 7.1052182
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.86
Output dim: 6, lower bound: -7.1051752, upper bound: 7.1051752
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.86
Output dim: 6, lower bound: -7.1051752, upper bound: 7.1051752

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.4072270, 1.6839716, -1.5494717, 1.1625860, -3.5698130, 3.2334433
1: -1.7135484, 1.8749424, -1.1842891, 1.2255725, -2.9391208, 3.0592315
2: -2.3168027, 2.0132427, -1.4262667, 1.4593599, -3.7761626, 3.4395094
3: -2.8167443, 1.5214396, -1.7218329, 1.0459522, -3.8626966, 3.2432723
4: -2.9309113, 2.1248298, -1.8217064, 1.5358458, -4.4667568, 3.9465361
5: -2.4719813, 1.7053752, -1.6001296, 1.2347190, -3.7067003, 3.3055048
6: -2.7902274, 2.0556426, -1.6256394, 1.6346729, -4.4249001, 3.6812820
7: -2.2715850, 2.2815056, -1.5026498, 1.5290047, -3.8005896, 3.7841554
8: -3.1167779, 1.8052709, -1.8960295, 1.3308468, -4.4476247, 3.7013004
9: -2.1877351, 2.2236533, -1.4437523, 1.4995078, -3.6872430, 3.6674056

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0954903, upper bound: 7.0937974
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0945620, upper bound: 7.0933938
time: 3.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2.8868761, 2.0215201, -2.8998618, 2.0303218, -4.9171982, 4.9213820
1: -2.0103028, 2.2485301, -2.0186582, 2.2579587, -4.2682614, 4.2671881
2: -2.8274336, 2.3321276, -2.8410997, 2.3406897, -5.1681232, 5.1732273
3: -3.4302864, 1.7998930, -3.4468539, 1.8072621, -5.2375484, 5.2467470
4: -3.5837672, 2.4534206, -3.6010733, 2.4621482, -6.0459156, 6.0544939
5: -2.9871774, 1.9839300, -3.0008640, 1.9911168, -4.9782944, 4.9847941
6: -3.4335582, 2.3261924, -3.4494038, 2.3334591, -5.7670174, 5.7755961
7: -2.7062693, 2.7101002, -2.7184162, 2.7211928, -5.4274621, 5.4285164
8: -3.8118935, 2.1036921, -3.8297253, 2.1101062, -5.9219999, 5.9334173
9: -2.6123528, 2.6450043, -2.6237617, 2.6556935, -5.2680464, 5.2687659

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0958838, upper bound: 7.0939673
time: 5.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0948504, upper bound: 7.0935599
time: 4.23 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.0487289, 3.5503845, -1.5918683, 1.1859858, -6.2347145, 5.1422529
1: -3.4977493, 3.8182106, -1.2095428, 1.2564939, -4.7542434, 5.0277534
2: -5.0042419, 3.7574544, -1.4674232, 1.4866958, -6.4909377, 5.2248774
3: -6.0244641, 3.0168278, -1.7758467, 1.0693738, -7.0938377, 4.7926745
4: -6.3126898, 3.8887734, -1.8728570, 1.5629734, -7.8756633, 5.7616305
5: -5.2742410, 3.2545311, -1.6421297, 1.2565114, -6.5307522, 4.8966608
6: -6.1592326, 3.5981421, -1.6825907, 1.6504616, -7.8096943, 5.2807331
7: -4.5627718, 4.5558515, -1.5398366, 1.5638564, -6.1266279, 6.0956879
8: -6.7332387, 3.4662037, -1.9548558, 1.3526418, -8.0858803, 5.4210596
9: -4.4136982, 4.4467707, -1.4800632, 1.5335611, -5.9472594, 5.9268341

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949360, upper bound: 7.0937446
time: 4.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0932534, upper bound: 7.0932550
time: 4.10 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.5790176, 3.9186964, -2.9761438, 2.0841699, -7.6631875, 6.8948402
1: -3.8853827, 4.1968937, -2.0653756, 2.3159854, -6.2013683, 6.2622690
2: -5.5349264, 4.1065722, -2.9208913, 2.3916783, -7.9266047, 7.0274634
3: -6.6628265, 3.3084962, -3.5427623, 1.8521789, -8.5150051, 6.8512583
4: -6.9720016, 4.2323856, -3.7027237, 2.5123923, -9.4843941, 7.9351091
5: -5.8345423, 3.5683770, -3.0871408, 2.0333753, -7.8679175, 6.6555176
6: -6.8209810, 3.8882303, -3.5494311, 2.3705559, -9.1915369, 7.4376612
7: -5.0162530, 5.0131140, -2.7871070, 2.7896183, -7.8058710, 7.8002210
8: -7.4437227, 3.7892971, -3.9382858, 2.1538634, -9.5975857, 7.7275829
9: -4.8498688, 4.8934441, -2.6909122, 2.7220440, -7.5719128, 7.5843563

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953987, upper bound: 7.0939051
time: 3.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0934024, upper bound: 7.0934024
time: 2.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 7.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0954903, upper bound: 7.0937974
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0945620, upper bound: 7.0933938
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0958838, upper bound: 7.0939673
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0948504, upper bound: 7.0935599
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0949360, upper bound: 7.0937446
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0932534, upper bound: 7.0932550
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0953987, upper bound: 7.0939051
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.53
Output dim: 6, lower bound: -7.0934024, upper bound: 7.0934024

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.7388140, 1.2665004, -1.0868421, 0.9122059, -2.6510201, 2.3533425
1: -1.2979916, 1.3685390, -0.9185759, 0.8877735, -2.1857653, 2.2871149
2: -1.6111245, 1.5812814, -0.9819836, 1.1638894, -2.7750139, 2.5632651
3: -1.9571488, 1.1507071, -1.1188666, 0.7981334, -2.7552822, 2.2695737
4: -2.0528865, 1.6751211, -1.2558689, 1.2305622, -3.2834487, 2.9309900
5: -1.7917237, 1.3330643, -1.1436284, 0.9955104, -2.7872341, 2.4766927
6: -1.9000161, 1.7185946, -0.9884387, 1.4726759, -3.3726921, 2.7070332
7: -1.6667379, 1.6965859, -1.1098875, 1.1472576, -2.8139954, 2.8064733
8: -2.1634834, 1.4175749, -1.3037356, 1.0775465, -3.2410297, 2.7213106
9: -1.6038060, 1.6535029, -1.0548499, 1.1383171, -2.7421231, 2.7083528

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953577, upper bound: 7.0937974
time: 3.78 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0954903, upper bound: 7.0937974
time: 3.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.8489809, 3.3212128, -1.0546396, 0.8949430, -5.7439241, 4.3758526
1: -3.2190773, 3.7192507, -0.8983164, 0.8670015, -4.0860786, 4.6175671
2: -4.8667607, 3.6019893, -0.9514644, 1.1422764, -6.0090370, 4.5534534
3: -5.8995619, 2.9291184, -1.0754824, 0.7813327, -6.6808949, 4.0046005
4: -6.2268491, 3.7810614, -1.2148629, 1.2081668, -7.4350157, 4.9959245
5: -4.9518442, 3.1019044, -1.1124291, 0.9788398, -5.9306841, 4.2143335
6: -5.9766884, 3.4443202, -0.9431797, 1.4614302, -7.4381185, 4.3874998
7: -4.4518108, 4.4006290, -1.0826256, 1.1215817, -5.5733924, 5.4832544
8: -6.5773921, 3.2572286, -1.2631295, 1.0588909, -7.6362829, 4.5203581
9: -4.3056583, 4.3257928, -1.0269399, 1.1133637, -5.4190221, 5.3527327

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0944068, upper bound: 7.0933937
time: 3.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0945620, upper bound: 7.0933937
time: 3.76 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2.1783347, 1.5340524, -2.3832152, 1.6704355, -3.8487701, 3.9172676
1: -1.5690416, 1.7037385, -1.6974690, 1.8604723, -3.4295139, 3.4012074
2: -2.0718126, 1.8651034, -2.2916622, 1.9988111, -4.0706239, 4.1567655
3: -2.5188339, 1.3922925, -2.7845533, 1.5084265, -4.0272603, 4.1768456
4: -2.6156909, 1.9717236, -2.8989429, 2.1117067, -4.7273979, 4.8706665
5: -2.2383027, 1.5767264, -2.4494042, 1.6944990, -3.9328017, 4.0261307
6: -2.4971399, 1.9370942, -2.7686493, 2.0475550, -4.5446949, 4.7057438
7: -2.0599949, 2.0825808, -2.2484043, 2.2632692, -4.3232641, 4.3309851
8: -2.7893670, 1.6824867, -3.0861475, 1.7995148, -4.5888815, 4.7686343
9: -1.9858581, 2.0246184, -2.1673741, 2.2046752, -4.1905332, 4.1919928

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0958195, upper bound: 7.0939673
time: 3.20 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0958838, upper bound: 7.0939673
time: 5.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.3463211, 3.7721734, -2.3544545, 1.6495225, -6.9958439, 6.1266279
1: -3.5132222, 4.1076231, -1.6778769, 1.8389232, -5.3521452, 5.7855000
2: -5.3690004, 3.9539485, -2.2592316, 1.9800203, -7.3490210, 6.2131801
3: -6.5023494, 3.2326591, -2.7458639, 1.4925447, -7.9948940, 5.9785233
4: -6.8581090, 4.1092019, -2.8584528, 2.0911689, -8.9492779, 6.9676547
5: -5.7114329, 3.3738842, -2.4201682, 1.6777297, -7.3891625, 5.7940521
6: -6.6181030, 3.7164567, -2.7321610, 2.0290916, -8.6471949, 6.4486179
7: -4.8896165, 4.8756838, -2.2209597, 2.2385693, -7.1281857, 7.0966434
8: -7.2681794, 3.5976562, -3.0439887, 1.7815564, -9.0497360, 6.6416450
9: -4.7400651, 4.7373419, -2.1415560, 2.1792457, -6.9193106, 6.8788977

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0947399, upper bound: 7.0935599
time: 3.10 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0948504, upper bound: 7.0935599
time: 3.69 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2519603, 3.0000570, -1.1249855, 0.9326152, -5.1845756, 4.1250424
1: -2.9153233, 3.2564182, -0.9414652, 0.9132845, -3.8286078, 4.1978836
2: -4.2099152, 3.2379637, -1.0168021, 1.1890609, -5.3989763, 4.2547655
3: -5.0682621, 2.5761976, -1.1691408, 0.8186439, -5.8869061, 3.7453384
4: -5.3305421, 3.3752275, -1.3040308, 1.2545995, -6.5851417, 4.6792583
5: -4.4440818, 2.7864618, -1.1811252, 1.0145841, -5.4586658, 3.9675870
6: -5.1818085, 3.1641226, -1.0422940, 1.4831871, -6.6649957, 4.2064166
7: -3.8833320, 3.8753724, -1.1406727, 1.1780341, -5.0613661, 5.0160451
8: -5.6815615, 2.9842036, -1.3520112, 1.0965540, -6.7781153, 4.3362150
9: -3.7599788, 3.7846332, -1.0872676, 1.1679813, -4.9279599, 4.8719006

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0948783, upper bound: 7.0937446
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0949360, upper bound: 7.0937446
time: 3.79 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.5808845, 5.2846870, -1.0851151, 0.9116455, -8.4925299, 6.3698020
1: -5.3327188, 5.6027303, -0.9170740, 0.8867314, -6.2194505, 6.5198045
2: -7.5102916, 5.4066525, -0.9796480, 1.1627982, -8.6730900, 6.3863006
3: -9.0545635, 4.4113202, -1.1162046, 0.7980039, -9.8525677, 5.5275249
4: -9.4628105, 5.5101252, -1.2542286, 1.2271459, -10.6899567, 6.7643538
5: -7.9228706, 4.7377863, -1.1429306, 0.9942938, -8.9171648, 5.8807168
6: -9.2627373, 5.0058937, -0.9860630, 1.4697468, -10.7324839, 5.9919567
7: -6.7086878, 6.7107654, -1.1075716, 1.1465530, -7.8552408, 7.8183370
8: -10.0883751, 4.9582968, -1.3026127, 1.0736718, -11.1620464, 6.2609096
9: -6.4722457, 6.5521760, -1.0533574, 1.1374372, -7.6096830, 7.6055336

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0932140, upper bound: 7.0932550
time: 2.46 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0932534, upper bound: 7.0932550
time: 2.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.7484312, 3.3454311, -2.4560804, 1.7183734, -6.4668045, 5.8015118
1: -3.2793741, 3.6116176, -1.7428664, 1.9164083, -5.1957827, 5.3544841
2: -4.7075605, 3.5646169, -2.3691866, 2.0469389, -6.7544994, 5.9338036
3: -5.6621222, 2.8491678, -2.8784368, 1.5503975, -7.2125196, 5.7276049
4: -5.9484110, 3.6973441, -2.9991980, 2.1602969, -8.1087074, 6.6965418
5: -4.9688187, 3.0806060, -2.5254450, 1.7355354, -6.7043543, 5.6060510
6: -5.7996554, 3.4392557, -2.8658428, 2.0829263, -7.8825817, 6.3050985
7: -4.3082285, 4.3031960, -2.3150601, 2.3281524, -6.6363811, 6.6182561
8: -6.3483086, 3.2890146, -3.1925693, 1.8395096, -8.1878185, 6.4815836
9: -4.1683822, 4.2023597, -2.2318695, 2.2696810, -6.4380631, 6.4342289

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953788, upper bound: 7.0939052
time: 4.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0953987, upper bound: 7.0939051
time: 4.81 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.0760641, 5.6275444, -2.4213817, 1.6932774, -9.7693415, 8.0489264
1: -5.6954403, 5.9547205, -1.7195829, 1.8902016, -7.5856419, 7.6743035
2: -8.0052433, 5.7320352, -2.3305180, 2.0242019, -10.0294456, 8.0625534
3: -9.6511984, 4.6825624, -2.8320222, 1.5311556, -11.1823540, 7.5145845
4: -10.0813103, 5.8321233, -2.9502296, 2.1356928, -12.2170029, 8.7823524
5: -8.4450855, 5.0304356, -2.4901078, 1.7152838, -10.1603689, 7.5205431
6: -9.8795357, 5.2849679, -2.8213687, 2.0607641, -11.9403000, 8.1063366
7: -7.1322942, 7.1360393, -2.2821109, 2.2982955, -9.4305897, 9.4181499
8: -10.7520533, 5.2606111, -3.1417856, 1.8179252, -12.5699787, 8.4023972
9: -6.8785830, 6.9694672, -2.2009153, 2.2390404, -9.1176233, 9.1703825

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0933839, upper bound: 7.0934024
time: 2.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0934024, upper bound: 7.0934024
time: 2.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.49 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0953577, upper bound: 7.0937974
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0954903, upper bound: 7.0937974
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0944068, upper bound: 7.0933937
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0945620, upper bound: 7.0933937
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0958195, upper bound: 7.0939673
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0958838, upper bound: 7.0939673
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0947399, upper bound: 7.0935599
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0948504, upper bound: 7.0935599
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0948783, upper bound: 7.0937446
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0949360, upper bound: 7.0937446
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0932140, upper bound: 7.0932550
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0932534, upper bound: 7.0932550
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0953788, upper bound: 7.0939052
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0953987, upper bound: 7.0939051
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0933839, upper bound: 7.0934024
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.49
Output dim: 6, lower bound: -7.0934024, upper bound: 7.0934024

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8559082, 0.7950900, -0.4318364, 0.5166551, -1.3725634, 1.2269263
1: -0.7781308, 0.7556261, -0.4659350, 0.4984617, -1.2765925, 1.2215611
2: -0.7872721, 1.0148118, -0.4638294, 0.6654212, -1.4526933, 1.4786412
3: -0.8243742, 0.6883797, -0.4412923, 0.4230151, -1.2473893, 1.1296721
4: -0.9803509, 1.0967195, -0.5178279, 0.7401119, -1.7204628, 1.6145474
5: -0.9324714, 0.8825681, -0.5608130, 0.5960482, -1.5285196, 1.4433811
6: -0.7068985, 1.4122058, -0.1212386, 1.3249648, -2.0318632, 1.5334444
7: -0.9277446, 0.9748831, -0.6111265, 0.5566871, -1.4844317, 1.5860096
8: -1.0399210, 0.9448092, -0.5584123, 0.6839261, -1.7238472, 1.5032215
9: -0.8693105, 0.9735827, -0.5317736, 0.5557728, -1.4250834, 1.5053563

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0917226, upper bound: 7.0880633
time: 5.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909670, upper bound: 7.0879008
time: 2.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.4319140, 1.0971951, -0.7492908, 0.7346572, -2.1665711, 1.8464860
1: -1.1150831, 1.1413965, -0.7139849, 0.6974384, -1.8125215, 1.8553815
2: -1.3055930, 1.3877020, -0.7156734, 0.9294664, -2.2350595, 2.1033754
3: -1.5626997, 0.9872638, -0.7186914, 0.6262324, -2.1889319, 1.7059553
4: -1.6778876, 1.4740415, -0.8703553, 1.0118748, -2.6897624, 2.3443968
5: -1.4850676, 1.1739404, -0.8407597, 0.8206633, -2.3057308, 2.0147002
6: -1.4881124, 1.5895156, -0.5542101, 1.3949339, -2.8830464, 2.1437256
7: -1.3975637, 1.4400225, -0.8477790, 0.8778939, -2.2754576, 2.2878015
8: -1.7464765, 1.2471306, -0.9170387, 0.8982929, -2.6447694, 2.1641693
9: -1.3427486, 1.4071715, -0.7867929, 0.8849155, -2.2276642, 2.1939645

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0920455, upper bound: 7.0880633
time: 3.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0911907, upper bound: 7.0879008
time: 5.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.8208129, 2.6317534, -0.4268151, 0.5134928, -4.3343058, 3.0585685
1: -2.5808172, 2.9338217, -0.4619020, 0.4950120, -3.0758293, 3.3957236
2: -3.7775707, 2.9281566, -0.4594944, 0.6614960, -4.4390669, 3.3876510
3: -4.5830097, 2.3406906, -0.4383359, 0.4199009, -5.0029106, 2.7790265
4: -4.8189421, 3.0736191, -0.5124910, 0.7343860, -5.5533280, 3.5861101
5: -3.8962023, 2.5134094, -0.5569340, 0.5917975, -4.4879999, 3.0703435
6: -4.6172080, 2.8660257, -0.1132844, 1.3222194, -5.9394274, 2.9793100
7: -3.5197823, 3.4952714, -0.6069713, 0.5521061, -4.0718884, 4.1022425
8: -5.0971107, 2.6595769, -0.5542026, 0.6783714, -5.7754822, 3.2137794
9: -3.4020250, 3.4235973, -0.5274179, 0.5507132, -3.9527383, 3.9510152

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0905262, upper bound: 7.0877171
time: 3.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900064, upper bound: 7.0876064
time: 5.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.5243373, 3.1033564, -0.7232347, 0.7199860, -5.2443233, 3.8265910
1: -3.0177758, 3.4709842, -0.6968230, 0.6816502, -3.6994259, 4.1678071
2: -4.5226183, 3.3892057, -0.6948639, 0.9081676, -5.4307861, 4.0840697
3: -5.4836955, 2.7431347, -0.6917454, 0.6117678, -6.0954633, 3.4348803
4: -5.7826495, 3.5581350, -0.8416292, 0.9927473, -6.7753968, 4.3997641
5: -4.6179790, 2.9161122, -0.8187560, 0.8035836, -5.4215627, 3.7348680
6: -5.5463147, 3.2628727, -0.5208449, 1.3875113, -6.9338260, 3.7837176
7: -4.1576719, 4.1143484, -0.8276516, 0.8536544, -5.0113263, 4.9419999
8: -6.1090579, 3.0677578, -0.8859398, 0.8830972, -6.9921551, 3.9536977
9: -4.0201445, 4.0404692, -0.7668766, 0.8618565, -4.8820009, 4.8073459

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
time: 3.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902369, upper bound: 7.0876064
time: 3.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.2378039, 0.9937134, -1.4807183, 1.1208181, -2.3586221, 2.4744315
1: -1.0027373, 1.0011405, -1.1449870, 1.1768873, -2.1796246, 2.1461275
2: -1.1147784, 1.2663043, -1.3509034, 1.4156305, -2.5304089, 2.6172075
3: -1.3086395, 0.8846506, -1.6199532, 1.0098685, -2.3185081, 2.5046039
4: -1.4369581, 1.3484808, -1.7328867, 1.5121919, -2.9491501, 3.0813675
5: -1.2948332, 1.0791968, -1.5282102, 1.2017833, -2.4966164, 2.6074071
6: -1.2318954, 1.5204170, -1.5576463, 1.6285985, -2.8604939, 3.0780632
7: -1.2285569, 1.2781912, -1.4370884, 1.4750384, -2.7035952, 2.7152796
8: -1.4980853, 1.1592374, -1.8032265, 1.2901733, -2.7882586, 2.9624639
9: -1.1795869, 1.2588487, -1.3803548, 1.4406285, -2.6202154, 2.6392035

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0919416, upper bound: 7.0880633
time: 4.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0912527, upper bound: 7.0879008
time: 4.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.8525465, 1.3311336, -1.9184554, 1.3691168, -3.2216632, 3.2495890
1: -1.3674589, 1.4555081, -1.4086990, 1.5032572, -2.8707161, 2.8642073
2: -1.7270850, 1.6541193, -1.7956630, 1.6956799, -3.4227648, 3.4497824
3: -2.0999234, 1.2117193, -2.1859550, 1.2461170, -3.3460402, 3.3976743
4: -2.1903536, 1.7521418, -2.2719922, 1.7955943, -3.9859481, 4.0241342
5: -1.9059002, 1.3957293, -1.9714804, 1.4301808, -3.3360810, 3.3672097
6: -2.0608859, 1.7766232, -2.1464467, 1.8080149, -3.8689008, 3.9230700
7: -1.7647145, 1.7943364, -1.8251410, 1.8498361, -3.6145506, 3.6194773
8: -2.3242092, 1.4951611, -2.4165640, 1.5308299, -3.8550391, 3.9117250
9: -1.6992815, 1.7475349, -1.7558179, 1.8017249, -3.5010064, 3.5033526

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0921938, upper bound: 7.0880633
time: 4.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0913957, upper bound: 7.0879008
time: 5.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.3161802, 3.0361786, -1.4537435, 1.1064389, -5.4226189, 4.4899220
1: -2.8800735, 3.3194118, -1.1287878, 1.1571409, -4.0372143, 4.4481993
2: -4.2894001, 3.2684112, -1.3227562, 1.3989811, -5.6883812, 4.5911674
3: -5.1974993, 2.6351030, -1.5840967, 0.9961327, -6.1936321, 4.2191997
4: -5.4681034, 3.4073269, -1.7001944, 1.4943182, -6.9624214, 5.1075211
5: -4.5548215, 2.7906916, -1.5023414, 1.1880398, -5.7428613, 4.2930331
6: -5.2625365, 3.1432204, -1.5239999, 1.6164637, -6.8790002, 4.6672201
7: -3.9606352, 3.9541223, -1.4117982, 1.4538171, -5.4144526, 5.3659205
8: -5.7988453, 2.9831905, -1.7691345, 1.2743704, -7.0732155, 4.7523251
9: -3.8357973, 3.8446388, -1.3566555, 1.4200876, -5.2558851, 5.2012944

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0905328, upper bound: 7.0877171
time: 4.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900271, upper bound: 7.0876064
time: 3.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.0098724, 3.5314894, -1.8860428, 1.3501985, -6.3600712, 5.4175320
1: -3.3066924, 3.8498380, -1.3879073, 1.4802728, -4.7869654, 5.2377453
2: -5.0161476, 3.7299628, -1.7610090, 1.6755993, -6.6917467, 5.4909716
3: -6.0760059, 3.0372598, -2.1431379, 1.2296621, -7.3056679, 5.1803980
4: -6.4043341, 3.8803515, -2.2320514, 1.7738979, -8.1782322, 6.1124029
5: -5.3329449, 3.1833923, -1.9399469, 1.4133537, -6.7462988, 5.1233392
6: -6.1752195, 3.5301838, -2.1059995, 1.7914878, -7.9667072, 5.6361833
7: -4.5863204, 4.5741854, -1.7950009, 1.8237098, -6.4100304, 6.3691864
8: -6.7875724, 3.3961911, -2.3720505, 1.5114975, -8.2990704, 5.7682419
9: -4.4445014, 4.4452915, -1.7279196, 1.7753928, -6.2198944, 6.1732111

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
time: 4.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902410, upper bound: 7.0876064
time: 3.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.1400704, 2.2264225, -0.4461668, 0.5291561, -3.6692266, 2.6725893
1: -2.1655269, 2.4486575, -0.4789988, 0.5074518, -2.6729786, 2.9276564
2: -3.0834968, 2.5112677, -0.4768036, 0.6783757, -3.7618725, 2.9880712
3: -3.7089670, 1.9589833, -0.4522879, 0.4332049, -4.1421719, 2.4112711
4: -3.9057307, 2.6547866, -0.5349424, 0.7539695, -4.6597004, 3.1897290
5: -3.2470396, 2.1423812, -0.5747490, 0.6076676, -3.8547072, 2.7171302
6: -3.7859702, 2.5655642, -0.1397101, 1.3276126, -5.1135826, 2.7052743
7: -2.9151874, 2.9128778, -0.6226184, 0.5726451, -3.4878325, 3.5354962
8: -4.1688328, 2.3310957, -0.5725605, 0.6969912, -4.8658237, 2.9036562
9: -2.8268478, 2.8501821, -0.5448853, 0.5746720, -3.4015198, 3.3950675

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0911726, upper bound: 7.0880169
time: 3.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900133, upper bound: 7.0878308
time: 3.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.8994663, 2.7562175, -0.7772474, 0.7510461, -4.6505122, 3.5334649
1: -2.6564353, 3.0051725, -0.7322474, 0.7131387, -3.3695741, 3.7374198
2: -3.8572035, 3.0067449, -0.7362303, 0.9515178, -4.8087215, 3.7429752
3: -4.6430774, 2.3832948, -0.7473524, 0.6418028, -5.2848802, 3.1306472
4: -4.8885040, 3.1505551, -0.9019374, 1.0293422, -5.9178462, 4.0524926
5: -4.0715981, 2.5792031, -0.8645092, 0.8373315, -4.9089298, 3.4437122
6: -4.7465429, 2.9763024, -0.5885435, 1.4006174, -6.1471605, 3.5648460
7: -3.5807393, 3.5714636, -0.8689424, 0.9027804, -4.4835196, 4.4404058
8: -5.2101970, 2.7757535, -0.9502324, 0.9120918, -6.1222887, 3.7259860
9: -3.4694369, 3.4886146, -0.8089706, 0.9085020, -4.3779387, 4.2975850

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0914093, upper bound: 7.0880169
time: 4.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900986, upper bound: 7.0878308
time: 3.95 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.4254069, 4.4877520, -0.4383229, 0.5235466, -6.9489536, 4.9260750
1: -4.4852386, 4.7811885, -0.4715888, 0.5025210, -4.9877596, 5.2527771
2: -6.3526616, 4.6497388, -0.4699657, 0.6718695, -7.0245309, 5.1197042
3: -7.6541700, 3.7792587, -0.4466716, 0.4284411, -8.0826111, 4.2259302
4: -8.0121202, 4.7619820, -0.5258820, 0.7453831, -8.7575035, 5.2878637
5: -6.7041931, 4.0568962, -0.5676565, 0.6013516, -7.3055449, 4.6245527
6: -7.8286886, 4.3647423, -0.1269457, 1.3240485, -9.1527367, 4.4916878
7: -5.7154412, 5.7151403, -0.6162249, 0.5650179, -6.2804589, 6.3313651
8: -8.5449619, 4.2796211, -0.5646515, 0.6889001, -9.2338619, 4.8442726
9: -5.5216355, 5.5781150, -0.5384005, 0.5658690, -6.0875044, 6.1165156

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
time: 3.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874486, upper bound: 7.0874811
time: 2.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2248769, 5.0390358, -0.7438558, 0.7327623, -7.9576392, 5.7828918
1: -5.0716796, 5.3494616, -0.7110226, 0.6940910, -5.7657704, 6.0604839
2: -7.1535201, 5.1734314, -0.7112865, 0.9253575, -8.0788774, 5.8847179
3: -8.6232376, 4.2164540, -0.7128765, 0.6238101, -9.2470474, 4.9293303
4: -9.0163536, 5.2802553, -0.8653176, 1.0064868, -10.0228405, 6.1455731
5: -7.5467596, 4.5280666, -0.8370620, 0.8171030, -8.3638630, 5.3651285
6: -8.8211651, 4.8096633, -0.5459502, 1.3916503, -10.2128153, 5.3556137
7: -6.4027967, 6.4036732, -0.8436291, 0.8737646, -7.2765613, 7.2473021
8: -9.6123161, 4.7486181, -0.9114515, 0.8932415, -10.5055580, 5.6600695
9: -6.1792517, 6.2517633, -0.7832879, 0.8812108, -7.0604625, 7.0350513

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
time: 3.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874779, upper bound: 7.0874811
time: 4.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6315637, 2.5746112, -1.5514838, 1.1602733, -4.7918367, 4.1260948
1: -2.4661117, 2.8148060, -1.1861323, 1.2294247, -3.6955364, 4.0009384
2: -3.5892265, 2.8338814, -1.4198154, 1.4613192, -5.0505457, 4.2536969
3: -4.3166699, 2.2381325, -1.7109022, 1.0482428, -5.3649130, 3.9490347
4: -4.5466223, 2.9803581, -1.8191636, 1.5575857, -6.1042080, 4.7995214
5: -3.7898014, 2.4240549, -1.5996118, 1.2381691, -5.0279703, 4.0236664
6: -4.4186440, 2.8413134, -1.6529546, 1.6546752, -6.0733194, 4.4942679
7: -3.3486013, 3.3402331, -1.4995332, 1.5341567, -4.8827581, 4.8397665
8: -4.8560638, 2.6341648, -1.9009154, 1.3272405, -6.1833043, 4.5350800
9: -3.2467473, 3.2664311, -1.4410709, 1.4984475, -4.7451949, 4.7075019

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0914723, upper bound: 7.0880169
time: 3.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0903186, upper bound: 7.0878308
time: 3.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.3804770, 3.0915000, -1.9964130, 1.4164214, -5.7968984, 5.0879130
1: -3.0094271, 3.3489668, -1.4568914, 1.5618863, -4.5713134, 4.8058581
2: -4.3385878, 3.3235297, -1.8778917, 1.7457829, -6.0843706, 5.2014213
3: -5.2200851, 2.6469207, -2.2862449, 1.2891937, -6.5092788, 4.9331656
4: -5.4891863, 3.4599969, -2.3679433, 1.8469353, -7.3361216, 5.8279400
5: -4.5804157, 2.8640089, -2.0519600, 1.4713190, -6.0517349, 4.9159689
6: -5.3443599, 3.2441692, -2.2505119, 1.8422465, -7.1866064, 5.4946814
7: -3.9923086, 3.9849191, -1.8954178, 1.9188054, -5.9111137, 5.8803368
8: -5.8570476, 3.0727417, -2.5280318, 1.5731010, -7.4301486, 5.6007738
9: -3.8648849, 3.8930039, -1.8248440, 1.8672466, -5.7321315, 5.7178478

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0916077, upper bound: 7.0880169
time: 4.26 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0903781, upper bound: 7.0878308
time: 3.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.9396496, 4.8444300, -1.5185010, 1.1424851, -8.0821342, 6.3629313
1: -4.8620534, 5.1474438, -1.1663308, 1.2052521, -6.0673056, 6.3137746
2: -6.8674688, 4.9881225, -1.3858726, 1.4406538, -8.3081226, 6.3739948
3: -8.2740545, 4.0609384, -1.6675520, 1.0312262, -9.3052807, 5.7284904
4: -8.6554556, 5.0966234, -1.7792910, 1.5356691, -10.1911249, 6.8759146
5: -7.2470598, 4.3614187, -1.5681113, 1.2211390, -8.4681988, 5.9295301
6: -8.4697618, 4.6553092, -1.6112490, 1.6394358, -10.1091976, 6.2665582
7: -6.1556082, 6.1569738, -1.4691420, 1.5081517, -7.6637597, 7.6261158
8: -9.2358866, 4.5935392, -1.8573159, 1.3079123, -10.5437984, 6.4508553
9: -5.9438262, 6.0121489, -1.4124891, 1.4724565, -7.4162827, 7.4246378

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
time: 3.09 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874570, upper bound: 7.0874811
time: 7.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.7102928, 5.3756433, -1.9577005, 1.3926122, -9.1029053, 7.3333435
1: -5.4273481, 5.6950846, -1.4319080, 1.5338939, -6.9612422, 7.1269927
2: -7.6393542, 5.4927802, -1.8359474, 1.7215159, -9.3608704, 7.3287277
3: -9.2086143, 4.4822965, -2.2353153, 1.2690752, -10.4776897, 6.7176118
4: -9.6235123, 5.5959339, -2.3203478, 1.8201395, -11.4436522, 7.9162817
5: -8.0593052, 4.8153677, -2.0135250, 1.4509084, -9.5102139, 6.8288927
6: -9.4262333, 5.0837007, -2.2012796, 1.8217169, -11.2479506, 7.2849803
7: -6.8184357, 6.8206296, -1.8598763, 1.8865857, -8.7050209, 8.6805058
8: -10.2644196, 5.0451441, -2.4746959, 1.5495744, -11.8139935, 7.5198402
9: -6.5777078, 6.6614261, -1.7909350, 1.8355343, -8.4132423, 8.4523611

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
time: 3.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874811, upper bound: 7.0874811
time: 3.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.81 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0917226, upper bound: 7.0880633
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0909670, upper bound: 7.0879008
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0920455, upper bound: 7.0880633
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0911907, upper bound: 7.0879008
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0905262, upper bound: 7.0877171
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0900064, upper bound: 7.0876064
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0902369, upper bound: 7.0876064
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0919416, upper bound: 7.0880633
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0912527, upper bound: 7.0879008
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0921938, upper bound: 7.0880633
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0913957, upper bound: 7.0879008
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0905328, upper bound: 7.0877171
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0900271, upper bound: 7.0876064
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0902410, upper bound: 7.0876064
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0911726, upper bound: 7.0880169
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0900133, upper bound: 7.0878308
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0914093, upper bound: 7.0880169
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0900986, upper bound: 7.0878308
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0874486, upper bound: 7.0874811
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0874779, upper bound: 7.0874811
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0914723, upper bound: 7.0880169
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0903186, upper bound: 7.0878308
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0916077, upper bound: 7.0880169
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0903781, upper bound: 7.0878308
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0874570, upper bound: 7.0874811
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.81
Output dim: 6, lower bound: -7.0874811, upper bound: 7.0874811

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5003304, 0.5763669, -0.3400598, 0.4171656, -0.9174960, 0.9164268
1: -0.5304963, 0.5465503, -0.3840181, 0.4266955, -0.9571918, 0.9305683
2: -0.5199442, 0.7347152, -0.3770401, 0.5653723, -1.0853164, 1.1117553
3: -0.4874553, 0.4754887, -0.3634589, 0.3455352, -0.8329905, 0.8389477
4: -0.5960155, 0.8300757, -0.4238730, 0.6076353, -1.2036507, 1.2539487
5: -0.6313052, 0.6551855, -0.4594353, 0.4945021, -1.1258073, 1.1146207
6: -0.2447758, 1.3368404, 0.0368890, 1.2974617, -1.5422375, 1.2999514
7: -0.6721379, 0.6436749, -0.5106893, 0.4663021, -1.1384400, 1.1543641
8: -0.6379104, 0.7324874, -0.4701423, 0.5588417, -1.1967521, 1.2026297
9: -0.5927444, 0.6496766, -0.4400182, 0.4437322, -1.0364765, 1.0896947

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0910163, upper bound: 7.0862440
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0917226, upper bound: 7.0880633
time: 7.51 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.7443106, 1.2194611, -0.3439631, 0.4217395, -2.1660502, 1.5634242
1: -1.3072650, 1.2303290, -0.3880565, 0.4300880, -1.7373531, 1.6183856
2: -1.4503859, 1.6024739, -0.3812205, 0.5699855, -2.0203714, 1.9836943
3: -1.8043745, 1.1269622, -0.3673632, 0.3490687, -2.1534431, 1.4943254
4: -1.8792193, 1.7700250, -0.4276733, 0.6146446, -2.4938641, 2.1976984
5: -1.6659468, 1.3398908, -0.4642467, 0.4994951, -2.1654420, 1.8041376
6: -1.8397973, 1.6712350, 0.0271239, 1.3000933, -3.1398907, 1.6441110
7: -1.5791963, 1.6553344, -0.5157086, 0.4704239, -2.0496202, 2.1710429
8: -1.9585154, 1.5023924, -0.4745403, 0.5667118, -2.5252271, 1.9769328
9: -1.5219114, 1.5999649, -0.4441018, 0.4485143, -1.9704257, 2.0440667

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0901061, upper bound: 7.0860656
time: 12.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909670, upper bound: 7.0879008
time: 3.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.8180196, 0.7752557, -0.5087017, 0.5772397, -1.3952594, 1.2839575
1: -0.7553634, 0.7365370, -0.5348070, 0.5464334, -1.3017968, 1.2713439
2: -0.7600378, 0.9881048, -0.5275273, 0.7337589, -1.4937966, 1.5156322
3: -0.7848418, 0.6685899, -0.4956977, 0.4742027, -1.2590444, 1.1642876
4: -0.9416997, 1.0728531, -0.6056480, 0.8200312, -1.7617309, 1.6785011
5: -0.9002873, 0.8629693, -0.6346777, 0.6571635, -1.5574508, 1.4976470
6: -0.6598542, 1.4051836, -0.2382888, 1.3402727, -2.0001268, 1.6434723
7: -0.9010066, 0.9446986, -0.6725799, 0.6425480, -1.5435545, 1.6172786
8: -1.0011660, 0.9220881, -0.6396054, 0.7469380, -1.7481040, 1.5616935
9: -0.8412493, 0.9451749, -0.5953538, 0.6498526, -1.4911020, 1.5405288

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0914445, upper bound: 7.0862440
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0920455, upper bound: 7.0880633
time: 6.43 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.6520844, 1.7315540, -0.5132958, 0.5797853, -3.2318697, 2.2448499
1: -1.8290459, 2.0418916, -0.5387050, 0.5490861, -2.3781319, 2.5805964
2: -2.4754336, 2.1343765, -0.5316750, 0.7368239, -3.2122574, 2.6660514
3: -3.0830624, 1.6169333, -0.4993440, 0.4763823, -3.5594447, 2.1162772
4: -3.1148000, 2.3358860, -0.6104455, 0.8247625, -3.9395623, 2.9463315
5: -2.6457338, 1.8244450, -0.6384540, 0.6604301, -3.3061640, 2.4628990
6: -3.1779242, 2.2218995, -0.2459234, 1.3432448, -4.5211687, 2.4678230
7: -2.4545097, 2.4315038, -0.6762025, 0.6462966, -3.1008062, 3.1077063
8: -3.2309313, 2.0529311, -0.6443557, 0.7523840, -3.9833152, 2.6972866
9: -2.3506806, 2.3163128, -0.5987535, 0.6536133, -3.0042939, 2.9150662

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902933, upper bound: 7.0860656
time: 5.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0911907, upper bound: 7.0879008
time: 6.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.1317580, 2.1687520, -0.3378263, 0.4148111, -3.5465691, 2.5065782
1: -2.1523423, 2.4064939, -0.3816420, 0.4245998, -2.5769422, 2.7881360
2: -3.0470331, 2.4759562, -0.3748116, 0.5627195, -3.6097527, 2.8507679
3: -3.7005377, 1.9464957, -0.3614948, 0.3434485, -4.0439863, 2.3079906
4: -3.8750391, 2.6037669, -0.4219188, 0.6029075, -4.4779468, 3.0256858
5: -3.1884429, 2.1181068, -0.4568230, 0.4915549, -3.6799979, 2.5749297
6: -3.7058656, 2.4844854, 0.0434321, 1.2951517, -5.0010176, 2.4410534
7: -2.8947444, 2.8885777, -0.5077164, 0.4638659, -3.3586104, 3.3962941
8: -4.1058445, 2.2570026, -0.4674685, 0.5545938, -4.6604385, 2.7244711
9: -2.7961779, 2.8185530, -0.4377009, 0.4410531, -3.2372310, 3.2562537

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0897545, upper bound: 7.0859143
time: 3.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0905262, upper bound: 7.0877171
time: 4.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3704014, 3.7039595, -0.3431381, 0.4212876, -5.7916889, 4.0470977
1: -3.5311379, 4.1122465, -0.3872204, 0.4292451, -3.9603829, 4.4994669
2: -5.3862891, 3.9513841, -0.3805670, 0.5691916, -5.9554806, 4.3319511
3: -6.4833946, 3.2578282, -0.3668722, 0.3484244, -6.8318191, 3.6247003
4: -6.8843484, 4.1592870, -0.4271609, 0.6126029, -7.4969511, 4.5864477
5: -5.4346423, 3.4283051, -0.4635304, 0.4984916, -5.9331341, 3.8918357
6: -6.6552505, 3.8657842, 0.0302078, 1.2984669, -7.9537172, 3.8355763
7: -4.8770614, 4.8126903, -0.5146947, 0.4695904, -5.3466516, 5.3273849
8: -7.2819080, 3.7150862, -0.4735309, 0.5650002, -7.8469081, 4.1886172
9: -4.7384696, 4.7360954, -0.4433619, 0.4477742, -5.1862440, 5.1794572

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0891405, upper bound: 7.0858027
time: 4.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900064, upper bound: 7.0876064
time: 4.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.8119302, 2.6250725, -0.4900148, 0.5641799, -4.3761101, 3.1150873
1: -2.5750661, 2.9260395, -0.5184234, 0.5347155, -3.1097817, 3.4444628
2: -3.7676005, 2.9218855, -0.5116317, 0.7194130, -4.4870138, 3.4335172
3: -4.5713987, 2.3356152, -0.4824651, 0.4638469, -5.0352454, 2.8180802
4: -4.8068953, 3.0669966, -0.5850387, 0.8029916, -5.6098871, 3.6520352
5: -3.8863721, 2.5077825, -0.6186609, 0.6436177, -4.5299897, 3.1264434
6: -4.6030240, 2.8606110, -0.2132407, 1.3337872, -5.9368114, 3.0738516
7: -3.5117576, 3.4870200, -0.6582686, 0.6245705, -4.1363282, 4.1452885
8: -5.0826311, 2.6521888, -0.6200246, 0.7319158, -5.8145471, 3.2722135
9: -3.3938913, 3.4149704, -0.5805764, 0.6300425, -4.0239339, 3.9955468

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902234, upper bound: 7.0859143
time: 4.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
time: 3.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.0396380, 4.1511960, -0.5001449, 0.5708580, -6.6104960, 4.6513410
1: -3.9464853, 4.6223221, -0.5272699, 0.5407462, -4.4872313, 5.1495919
2: -6.0947728, 4.3897495, -0.5202491, 0.7269640, -6.8217368, 4.9099989
3: -7.3399963, 3.6408222, -0.4898177, 0.4692834, -7.8092799, 4.1306400
4: -7.8019781, 4.6201296, -0.5962476, 0.8125013, -8.6144791, 5.2163773
5: -6.1200657, 3.8110077, -0.6275310, 0.6509330, -6.7709985, 4.4385386
6: -7.5414715, 4.2266173, -0.2284155, 1.3382872, -8.8797588, 4.4550328
7: -5.4826183, 5.4014282, -0.6660250, 0.6340237, -6.1166420, 6.0674534
8: -8.2448025, 4.1014457, -0.6307508, 0.7418638, -8.9866667, 4.7321963
9: -5.3263202, 5.3229656, -0.5884862, 0.6400369, -5.9663572, 5.9114518

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0893495, upper bound: 7.0858027
time: 3.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902369, upper bound: 7.0876064
time: 2.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6853803, 0.7013621, -0.9866638, 0.8583536, -1.5437338, 1.6880258
1: -0.6723551, 0.6635399, -0.8563871, 0.8307135, -1.5030687, 1.5199270
2: -0.6620309, 0.8849039, -0.8888474, 1.0998096, -1.7618406, 1.7737513
3: -0.6461993, 0.5974205, -0.9824566, 0.7496235, -1.3958228, 1.5798770
4: -0.7956937, 0.9858475, -1.1228962, 1.1916459, -1.9873395, 2.1087437
5: -0.7893863, 0.7840075, -1.0458891, 0.9509367, -1.7403231, 1.8298967
6: -0.5036628, 1.3814833, -0.8794529, 1.4601488, -1.9638116, 2.2609363
7: -0.8037733, 0.8267851, -1.0291352, 1.0723920, -1.8761653, 1.8559203
8: -0.8446526, 0.8692051, -1.1849453, 1.0214337, -1.8660862, 2.0541506
9: -0.7399622, 0.8322424, -0.9705780, 1.0621874, -1.8021495, 1.8028203

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0857072, upper bound: 7.0670387
time: 3.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0856292, upper bound: 7.0423673
time: 3.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -2.3012748, 1.5563613, -0.9896823, 0.8596400, -3.1609149, 2.5460436
1: -1.6268536, 1.7923932, -0.8585217, 0.8325688, -2.4594223, 2.6509149
2: -2.1546702, 1.9306334, -0.8916819, 1.1012700, -3.2559402, 2.8223152
3: -2.6595922, 1.4397992, -0.9861882, 0.7506221, -3.4102142, 2.4259875
4: -2.7092209, 2.1134918, -1.1261399, 1.1940013, -3.9032221, 3.2396317
5: -2.3087816, 1.6479870, -1.0484208, 0.9526658, -3.2614474, 2.6964078
6: -2.7441914, 2.0520649, -0.8838686, 1.4626547, -4.2068462, 2.9359336
7: -2.1094294, 2.1538594, -1.0315057, 1.0741295, -3.1835589, 3.1853652
8: -2.8249106, 1.8603594, -1.1880280, 1.0249077, -3.8498182, 3.0483875
9: -2.0536187, 2.0498438, -0.9728357, 1.0638013, -3.1174200, 3.0226793

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0823334, upper bound: 7.0669603
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0822983, upper bound: 7.0422016
time: 5.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.1357898, 0.9383522, -1.3765210, 1.0671828, -2.2029724, 2.3148732
1: -0.9441534, 0.9268957, -1.0834894, 1.1015592, -2.0457125, 2.0103850
2: -1.0208075, 1.2004608, -1.2517748, 1.3523237, -2.3731313, 2.4522357
3: -1.1765542, 0.8291578, -1.4893094, 0.9566258, -2.1331801, 2.3184671
4: -1.3117208, 1.2822672, -1.6074014, 1.4401326, -2.7518535, 2.8896685
5: -1.1928174, 1.0267379, -1.4290386, 1.1482561, -2.3410735, 2.4557767
6: -1.0881670, 1.4906015, -1.4176935, 1.5745898, -2.6627569, 2.9082952
7: -1.1485195, 1.1937389, -1.3469067, 1.3921748, -2.5406942, 2.5406456
8: -1.3712635, 1.1026272, -1.6733557, 1.2306638, -2.6019273, 2.7759829
9: -1.0956948, 1.1787157, -1.2945292, 1.3635659, -2.4592607, 2.4732449

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0916620, upper bound: 7.0862440
time: 5.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0921938, upper bound: 7.0880633
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.0736103, 2.0142856, -1.3822017, 1.0700192, -4.1436296, 3.3964872
1: -2.1161656, 2.3728392, -1.0871726, 1.1053898, -3.2215555, 3.4600120
2: -2.9743369, 2.4316545, -1.2573230, 1.3554326, -4.3297696, 3.6889775
3: -3.6326258, 1.8527598, -1.4962074, 0.9591529, -4.5917788, 3.3489671
4: -3.6636043, 2.6196785, -1.6138673, 1.4440422, -5.1076465, 4.2335458
5: -3.1013980, 2.0694141, -1.4341674, 1.1512997, -4.2526979, 3.5035815
6: -3.7469120, 2.4514041, -1.4257063, 1.5784082, -5.3253202, 3.8771105
7: -2.8570294, 2.8257885, -1.3516436, 1.3961525, -4.2531819, 4.1774321
8: -4.0313787, 2.3113995, -1.6796513, 1.2357886, -5.2671671, 3.9910507
9: -2.7268147, 2.7444441, -1.2989604, 1.3673707, -4.0941854, 4.0434046

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0905909, upper bound: 7.0860656
time: 5.49 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0913957, upper bound: 7.0879008
time: 5.20 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.5735397, 2.5051086, -0.9588109, 0.8446199, -4.4181595, 3.4639194
1: -2.4235361, 2.7500887, -0.8399093, 0.8141144, -3.2376504, 3.5899980
2: -3.5106440, 2.7736359, -0.8650761, 1.0816619, -4.5923061, 3.6387119
3: -4.2565022, 2.2042964, -0.9453961, 0.7368444, -4.9933467, 3.1496925
4: -4.4659252, 2.9034181, -1.0895486, 1.1739248, -5.6398501, 3.9929667
5: -3.7199869, 2.3697746, -1.0208311, 0.9371353, -4.6571221, 3.3906057
6: -4.2886844, 2.7355325, -0.8448488, 1.4523169, -5.7410011, 3.5803814
7: -3.2905047, 3.2889581, -1.0054591, 1.0519614, -4.3424664, 4.2944174
8: -4.7389970, 2.5394692, -1.1524870, 1.0071788, -5.7461758, 3.6919563
9: -3.1833847, 3.2003422, -0.9468883, 1.0434990, -4.2268839, 4.1472306

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0796012, upper bound: 7.0666624
time: 4.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0792806, upper bound: 7.0420504
time: 3.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.8492560, 4.1563487, -0.9786993, 0.8540474, -6.7033033, 5.1350479
1: -3.8110971, 4.4865909, -0.8517445, 0.8259754, -4.6370726, 5.3383355
2: -5.8657622, 4.2938719, -0.8811990, 1.0941684, -6.9599304, 5.1750708
3: -7.0582643, 3.5522807, -0.9706373, 0.7459626, -7.8042269, 4.5229177
4: -7.4834995, 4.4716043, -1.1124551, 1.1870942, -8.6705933, 5.5840597
5: -6.2074943, 3.6868267, -1.0388535, 0.9473124, -7.1548066, 4.7256804
6: -7.2650661, 4.1168389, -0.8713835, 1.4591480, -8.7242146, 4.9882226
7: -5.2954950, 5.2746553, -1.0214827, 1.0664350, -6.3619299, 6.2961378
8: -7.9438190, 4.0472536, -1.1751585, 1.0193384, -8.9631577, 5.2224121
9: -5.1567574, 5.1268616, -0.9629725, 1.0565238, -6.2132812, 6.0898342

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0719168, upper bound: 7.0666389
time: 21.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0717689, upper bound: 7.0419608
time: 3.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.2520733, 2.9897931, -1.3395455, 1.0478611, -5.2999344, 4.3293386
1: -2.8408537, 3.2690258, -1.0620756, 1.0744797, -3.9153333, 4.3311014
2: -4.2216010, 3.2251997, -1.2140760, 1.3296623, -5.5512633, 4.4392757
3: -5.1158333, 2.5976167, -1.4408801, 0.9378096, -6.0536427, 4.0384970
4: -5.3816252, 3.3637497, -1.5621552, 1.4159749, -6.7975998, 4.9259052
5: -4.4810505, 2.7540617, -1.3937142, 1.1302413, -5.6112919, 4.1477757
6: -5.1758251, 3.1090956, -1.3695803, 1.5590097, -6.7348347, 4.4786758
7: -3.9027112, 3.8953824, -1.3141464, 1.3625064, -5.2652178, 5.2095289
8: -5.7052102, 2.9437504, -1.6282768, 1.2092246, -6.9144349, 4.5720272
9: -3.7788422, 3.7878447, -1.2633088, 1.3361194, -5.1149616, 5.0511532

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902375, upper bound: 7.0859143
time: 4.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
time: 3.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.5319424, 4.6428394, -1.3675635, 1.0623859, -7.5943284, 6.0104027
1: -4.2305541, 5.0083957, -1.0782783, 1.0948212, -5.3253756, 6.0866737
2: -6.5812960, 4.7486982, -1.2414021, 1.3465621, -7.9278584, 5.9901004
3: -7.9232745, 3.9483802, -1.4765530, 0.9519160, -8.8751907, 5.4249334
4: -8.4060249, 4.9370651, -1.5959963, 1.4344022, -9.8404274, 6.5330615
5: -6.9730010, 4.0732565, -1.4204712, 1.1444324, -8.1174335, 5.4937277
6: -8.1634617, 4.4795170, -1.4080045, 1.5714979, -9.7349596, 5.8875217
7: -5.9102907, 5.8856711, -1.3377872, 1.3849695, -7.2952604, 7.2234583
8: -8.9174328, 4.4516349, -1.6621943, 1.2270373, -10.1444702, 6.1138291
9: -5.7562795, 5.7186766, -1.2861081, 1.3567469, -7.1130266, 7.0047846

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0893723, upper bound: 7.0858027
time: 3.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0902410, upper bound: 7.0876064
time: 3.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2.4064262, 1.7163146, -0.3470417, 0.4275461, -2.8339722, 2.0633564
1: -1.7091842, 1.8878458, -0.3917236, 0.4328359, -2.1420202, 2.2795694
2: -2.3113823, 2.0235751, -0.3851620, 0.5745288, -2.8859110, 2.4087372
3: -2.7774711, 1.5325962, -0.3710690, 0.3525969, -3.1300678, 1.9036653
4: -2.9147229, 2.1611297, -0.4313187, 0.6195403, -3.5342631, 2.5924485
5: -2.4566474, 1.7258060, -0.4689367, 0.5040828, -2.9607301, 2.1947427
6: -2.8244541, 2.1604688, 0.0235994, 1.2989432, -4.1233974, 2.1368694
7: -2.2503643, 2.2651174, -0.5200730, 0.4735692, -2.7239335, 2.7851903
8: -3.1159708, 1.8959682, -0.4778668, 0.5695007, -3.6854715, 2.3738351
9: -2.1829042, 2.2133646, -0.4482881, 0.4540645, -2.6369689, 2.6616526

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0904589, upper bound: 7.0861962
time: 2.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0911726, upper bound: 7.0880169
time: 3.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.4096966, 3.1615849, -0.3505571, 0.4316189, -4.8413153, 3.5121419
1: -2.9491553, 3.4007874, -0.3954107, 0.4359061, -3.3850615, 3.7961981
2: -4.3766880, 3.3715434, -0.3889261, 0.5786583, -4.9553461, 3.7604694
3: -5.2259555, 2.7241883, -0.3746189, 0.3557475, -5.5817032, 3.0988073
4: -5.5474448, 3.5386686, -0.4347496, 0.6259941, -6.1734390, 3.9734182
5: -4.6153007, 2.8783638, -0.4732840, 0.5085512, -5.1238518, 3.3516479
6: -5.4255114, 3.4558775, 0.0146528, 1.3016059, -6.7271175, 3.4412246
7: -4.0226932, 3.9857745, -0.5246704, 0.4772717, -4.4999647, 4.5104446
8: -5.9427691, 3.2367451, -0.4820429, 0.5768799, -6.5196490, 3.7187881
9: -3.9051008, 3.9022551, -0.4519732, 0.4583239, -4.3634248, 4.3542280

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0891376, upper bound: 7.0859957
time: 3.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900133, upper bound: 7.0878308
time: 3.12 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.1020880, 2.1988463, -0.5292110, 0.5921085, -3.6941965, 2.7280574
1: -2.1429417, 2.4198251, -0.5516166, 0.5598830, -2.7028246, 2.9714417
2: -3.0454831, 2.4855137, -0.5443649, 0.7506633, -3.7961464, 3.0298786
3: -3.6642923, 1.9347701, -0.5120736, 0.4878863, -4.1521788, 2.4468436
4: -3.8598657, 2.6313004, -0.6286176, 0.8371162, -4.6969819, 3.2599182
5: -3.2053075, 2.1212785, -0.6518519, 0.6715611, -3.8768687, 2.7731304
6: -3.7374349, 2.5455506, -0.2641578, 1.3447871, -5.0822220, 2.8097084
7: -2.8832204, 2.8805866, -0.6875898, 0.6628221, -3.5460424, 3.5681765
8: -4.1159110, 2.3043885, -0.6615194, 0.7602476, -4.8761587, 2.9659081
9: -2.7945812, 2.8188412, -0.6127934, 0.6720451, -3.4666262, 3.4316347

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0907272, upper bound: 7.0861962
time: 3.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0914093, upper bound: 7.0880169
time: 3.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2948189, 3.7463379, -0.5335948, 0.5942277, -5.8890467, 4.2799330
1: -3.6521609, 3.9954655, -0.5550406, 0.5622585, -4.2144194, 4.5505061
2: -5.2163434, 3.9406543, -0.5479436, 0.7533507, -5.9696941, 4.4885979
3: -6.2498260, 3.1792526, -0.5156958, 0.4899429, -6.7397690, 3.6949484
4: -6.5907316, 4.0719137, -0.6328065, 0.8414140, -7.4321456, 4.7047200
5: -5.4922233, 3.4208012, -0.6551467, 0.6743281, -6.1665516, 4.0759478
6: -6.4651041, 3.9079053, -0.2712846, 1.3474994, -7.8126035, 4.1791897
7: -4.7382779, 4.7209902, -0.6906819, 0.6660874, -5.4043655, 5.4116721
8: -7.0511880, 3.7509570, -0.6658652, 0.7651603, -7.8163481, 4.4168224
9: -4.5982704, 4.6115503, -0.6159006, 0.6752069, -5.2734776, 5.2274508

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0891764, upper bound: 7.0859957
time: 3.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0900986, upper bound: 7.0878308
time: 4.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.6283383, 3.9376702, -0.3435381, 0.4235388, -6.0518770, 4.2812085
1: -3.9019935, 4.2142053, -0.3879879, 0.4296203, -4.3316140, 4.6021934
2: -5.5553422, 4.1270027, -0.3815479, 0.5702732, -6.1256151, 4.5085506
3: -6.6895671, 3.3416193, -0.3678090, 0.3492977, -7.0388646, 3.7094283
4: -7.0146890, 4.2456756, -0.4281031, 0.6124676, -7.6271567, 4.6737785
5: -5.8664246, 3.5868461, -0.4646838, 0.4994280, -6.3658524, 4.0515299
6: -6.8389335, 3.9206357, 0.0330778, 1.2960215, -8.1349545, 3.8875580
7: -5.0332952, 5.0299039, -0.5153970, 0.4697739, -5.5030689, 5.5453010
8: -7.4811783, 3.8079374, -0.4737306, 0.5630773, -8.0442553, 4.2816677
9: -4.8664331, 4.9074564, -0.4445950, 0.4496430, -5.3160763, 5.3520513

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0881351, upper bound: 7.0857879
time: 3.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
time: 3.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.0497103, 5.6243744, -0.3485890, 0.4297076, -8.4794178, 5.9729633
1: -5.6483169, 5.9257507, -0.3933430, 0.4340537, -6.0823708, 6.3190937
2: -7.9473639, 5.7281237, -0.3870305, 0.5764377, -8.5238018, 6.1151543
3: -9.5744190, 4.7006216, -0.3729618, 0.3540176, -9.9284363, 5.0735836
4: -10.0009184, 5.8306227, -0.4331083, 0.6218066, -10.6227245, 6.2637310
5: -8.3622704, 5.0335827, -0.4710918, 0.5060184, -8.8682890, 5.5046744
6: -9.8242607, 5.3666635, 0.0204231, 1.2993675, -11.1236286, 5.3462405
7: -7.0720906, 7.0671611, -0.5221137, 0.4752212, -7.5473118, 7.5892749
8: -10.6917849, 5.3759937, -0.4796803, 0.5731174, -11.2649021, 5.8556743
9: -6.8345175, 6.9132280, -0.4499992, 0.4560445, -7.2905622, 7.3632274

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0861587, upper bound: 7.0856976
time: 2.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874486, upper bound: 7.0874811
time: 2.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.4006968, 4.4704866, -0.5055412, 0.5763063, -6.9770031, 4.9760280
1: -4.4685259, 4.7632089, -0.5323053, 0.5444955, -5.0130215, 5.2955141
2: -6.3289952, 4.6330862, -0.5248530, 0.7318596, -7.0608549, 5.1579390
3: -7.6255436, 3.7640204, -0.4937838, 0.4729217, -8.0984650, 4.2578044
4: -7.9849424, 4.7468777, -0.6032401, 0.8155451, -8.8004875, 5.3501177
5: -6.6790123, 4.0421720, -0.6328029, 0.6548110, -7.3338232, 4.6749749
6: -7.7979832, 4.3514185, -0.2320867, 1.3370879, -9.1350708, 4.5835052
7: -5.6960516, 5.6947641, -0.6699306, 0.6403962, -6.3364477, 6.3646946
8: -8.5122547, 4.2616420, -0.6368431, 0.7426271, -9.2548819, 4.8984852
9: -5.5016580, 5.5580888, -0.5935483, 0.6480639, -6.1497221, 6.1516371

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0884775, upper bound: 7.0857879
time: 3.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
time: 3.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8105097, 6.1486673, -0.5148675, 0.5824879, -9.3929977, 6.6635346
1: -6.2060223, 6.4663610, -0.5403513, 0.5501173, -6.7561398, 7.0067124
2: -8.7098351, 6.2265606, -0.5330480, 0.7386981, -9.4485331, 6.7596087
3: -10.4989567, 5.1168556, -0.5007922, 0.4779331, -10.9768896, 5.6176476
4: -10.9580059, 6.3239946, -0.6135402, 0.8244641, -11.7824697, 6.9375348
5: -9.1641254, 5.4821186, -0.6408665, 0.6615470, -9.8256721, 6.1229849
6: -10.7685051, 5.7892809, -0.2459616, 1.3413196, -12.1098251, 6.0352426
7: -7.7282271, 7.7227168, -0.6772040, 0.6489462, -8.3771734, 8.3999205
8: -11.7075701, 5.8219547, -0.6466606, 0.7518659, -12.4594364, 6.4686155
9: -7.4605770, 7.5550346, -0.6007605, 0.6573737, -8.1179504, 8.1557951

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0861916, upper bound: 7.0856976
time: 3.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874779, upper bound: 7.0874811
time: 3.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2.8418055, 2.0139122, -1.0528253, 0.8937263, -3.7355318, 3.0667377
1: -1.9797438, 2.2211699, -0.8960696, 0.8727890, -2.8525329, 3.1172395
2: -2.7698088, 2.3113353, -0.9472771, 1.1448133, -3.9146221, 3.2586124
3: -3.3286526, 1.7830290, -1.0684443, 0.7839023, -4.1125550, 2.8514733
4: -3.5019927, 2.4564619, -1.2081525, 1.2318439, -4.7338367, 3.6646144
5: -2.9107475, 1.9754293, -1.1102781, 0.9843774, -3.8951249, 3.0857074
6: -3.3995533, 2.4123130, -0.9702550, 1.4760029, -4.8755560, 3.3825679
7: -2.6421647, 2.6455371, -1.0832317, 1.1253214, -3.7674861, 3.7287688
8: -3.7455616, 2.1652856, -1.2691257, 1.0549655, -4.8005271, 3.4344113
9: -2.5636756, 2.5922804, -1.0265206, 1.1135125, -3.6771882, 3.6188011

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0908381, upper bound: 7.0861962
time: 4.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0914723, upper bound: 7.0880169
time: 5.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.9163647, 3.4681692, -1.0539067, 0.8941044, -5.8104692, 4.5220757
1: -3.2212632, 3.7296143, -0.8970569, 0.8734558, -4.0947189, 4.6266713
2: -4.8446846, 3.6596909, -0.9485185, 1.1449976, -5.9896822, 4.6082096
3: -5.7825737, 2.9708593, -1.0696778, 0.7839264, -6.5665002, 4.0405369
4: -6.1149926, 3.8344002, -1.2089679, 1.2329639, -7.3479567, 5.0433683
5: -5.1006765, 3.1833963, -1.1109698, 0.9851459, -6.0858226, 4.2943659
6: -5.9994850, 3.7016490, -0.9719151, 1.4781401, -7.4776249, 4.6735640
7: -4.4113965, 4.3729892, -1.0840290, 1.1254818, -5.5368786, 5.4570179
8: -6.5569806, 3.5251293, -1.2696514, 1.0575500, -7.6145306, 4.7947807
9: -4.2837687, 4.2756886, -1.0271734, 1.1137950, -5.3975639, 5.3028622

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0895570, upper bound: 7.0859957
time: 3.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0903186, upper bound: 7.0878308
time: 5.26 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.5179689, 2.4978578, -1.4499248, 1.1066331, -4.6246018, 3.9477825
1: -2.4006512, 2.7329552, -1.1256219, 1.1552957, -3.5559468, 3.8585773
2: -3.4772191, 2.7621622, -1.3230089, 1.3980180, -4.8752370, 4.0851712
3: -4.1832342, 2.1738634, -1.5831276, 0.9959428, -5.1791773, 3.7569909
4: -4.4082360, 2.9105952, -1.6964633, 1.4867493, -5.8949852, 4.6070585
5: -3.6693144, 2.3590815, -1.5020269, 1.1846368, -4.8539515, 3.8611083
6: -4.2790689, 2.7856860, -1.5167742, 1.6010294, -5.8800983, 4.3024602
7: -3.2538545, 3.2448406, -1.4114037, 1.4532850, -4.7071395, 4.6562443
8: -4.7043018, 2.5659690, -1.7658330, 1.2690194, -5.9733210, 4.3318019
9: -3.1534858, 3.1748338, -1.3570495, 1.4196372, -4.5731230, 4.5318832

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909903, upper bound: 7.0861962
time: 6.07 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0916077, upper bound: 7.0880169
time: 4.11 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.7447920, 4.0542922, -1.4533566, 1.1082792, -6.8530712, 5.5076489
1: -3.9798372, 4.3116932, -1.1280001, 1.1574643, -5.1373014, 5.4396935
2: -5.6733017, 4.2328629, -1.3263404, 1.3997347, -7.0730362, 5.5592031
3: -6.7828622, 3.4260261, -1.5871601, 0.9972318, -7.7800941, 5.0131865
4: -7.1469612, 4.3635674, -1.7001396, 1.4891756, -8.6361370, 6.0637069
5: -5.9649286, 3.6844285, -1.5049545, 1.1865450, -7.1514735, 5.1893830
6: -7.0210037, 4.1520085, -1.5215523, 1.6039636, -8.6249676, 5.6735611
7: -5.1261744, 5.1022048, -1.4141190, 1.4553216, -6.5814962, 6.5163240
8: -7.6549678, 4.0235734, -1.7692816, 1.2729571, -8.9279251, 5.7928553
9: -4.9632034, 4.9939351, -1.3595052, 1.4218059, -6.3850093, 6.3534403

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0895715, upper bound: 7.0859957
time: 4.13 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0903781, upper bound: 7.0878308
time: 3.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1027470, 4.2670412, -1.0189142, 0.8760573, -6.9788041, 5.2859554
1: -4.2494993, 4.5518270, -0.8753087, 0.8510084, -5.1005077, 5.4271355
2: -6.0300207, 4.4392419, -0.9158739, 1.1221634, -7.1521840, 5.3551159
3: -7.2608061, 3.6014214, -1.0230159, 0.7672789, -8.0280848, 4.6244373
4: -7.6079240, 4.5552511, -1.1646245, 1.2099522, -8.8178759, 5.7198753
5: -6.3658471, 3.8680050, -1.0786000, 0.9668341, -7.3326812, 4.9466052
6: -7.4304399, 4.1916265, -0.9260129, 1.4649646, -8.8954048, 5.1176395
7: -5.4384899, 5.4363298, -1.0546367, 1.0990782, -6.5375681, 6.4909668
8: -8.1181746, 4.0997519, -1.2266555, 1.0364169, -9.1545916, 5.3264074
9: -5.2553978, 5.3071952, -0.9976349, 1.0878309, -6.3432288, 6.3048301

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0881546, upper bound: 7.0857879
time: 3.27 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
time: 3.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.5186377, 5.9486408, -1.0374362, 0.8854741, -9.4041119, 6.9860773
1: -5.9921055, 6.2587867, -0.8866766, 0.8627785, -6.8548841, 7.1454635
2: -8.4167566, 6.0363002, -0.9322462, 1.1340249, -9.5507812, 6.9685464
3: -10.1422577, 4.9572406, -1.0469689, 0.7760401, -10.9182978, 6.0042095
4: -10.5889683, 6.1357174, -1.1877339, 1.2221626, -11.8111305, 7.3234510
5: -8.8573341, 5.3106365, -1.0958400, 0.9767184, -9.8340521, 6.4064765
6: -10.4077835, 5.6302805, -0.9517233, 1.4720266, -11.8798103, 6.5820036
7: -7.4751034, 7.4700890, -1.0697386, 1.1130235, -8.5881271, 8.5398273
8: -11.3211718, 5.6604629, -1.2492604, 1.0483408, -12.3695126, 6.9097233
9: -7.2194481, 7.3093181, -1.0128239, 1.1012642, -8.3207121, 8.3221416

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0862004, upper bound: 7.0856976
time: 3.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0874570, upper bound: 7.0874811
time: 3.15 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.23 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0910163, upper bound: 7.0862440
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0917226, upper bound: 7.0880633
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0901061, upper bound: 7.0860656
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0909670, upper bound: 7.0879008
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0914445, upper bound: 7.0862440
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0920455, upper bound: 7.0880633
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0902933, upper bound: 7.0860656
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0911907, upper bound: 7.0879008
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0897545, upper bound: 7.0859143
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0905262, upper bound: 7.0877171
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0891405, upper bound: 7.0858027
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0900064, upper bound: 7.0876064
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0902234, upper bound: 7.0859143
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0893495, upper bound: 7.0858027
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0902369, upper bound: 7.0876064
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0857072, upper bound: 7.0670387
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0856292, upper bound: 7.0423673
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0823334, upper bound: 7.0669603
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0822983, upper bound: 7.0422016
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0916620, upper bound: 7.0862440
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0921938, upper bound: 7.0880633
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0905909, upper bound: 7.0860656
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0913957, upper bound: 7.0879008
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0796012, upper bound: 7.0666624
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0792806, upper bound: 7.0420504
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0719168, upper bound: 7.0666389
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0717689, upper bound: 7.0419608
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0902375, upper bound: 7.0859143
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0909491, upper bound: 7.0877171
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0893723, upper bound: 7.0858027
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0902410, upper bound: 7.0876064
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0904589, upper bound: 7.0861962
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0911726, upper bound: 7.0880169
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0891376, upper bound: 7.0859957
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0900133, upper bound: 7.0878308
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0907272, upper bound: 7.0861962
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0914093, upper bound: 7.0880169
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0891764, upper bound: 7.0859957
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0900986, upper bound: 7.0878308
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0881351, upper bound: 7.0857879
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0861587, upper bound: 7.0856976
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0874486, upper bound: 7.0874811
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0884775, upper bound: 7.0857879
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0861916, upper bound: 7.0856976
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0874779, upper bound: 7.0874811
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0908381, upper bound: 7.0861962
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0914723, upper bound: 7.0880169
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0895570, upper bound: 7.0859957
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0903186, upper bound: 7.0878308
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0909903, upper bound: 7.0861962
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0916077, upper bound: 7.0880169
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0895715, upper bound: 7.0859957
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0903781, upper bound: 7.0878308
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0881546, upper bound: 7.0857879
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0889746, upper bound: 7.0875827
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0862004, upper bound: 7.0856976
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.23
Output dim: 6, lower bound: -7.0874570, upper bound: 7.0874811
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.23
Output dim: 6, lower bound: -7.0892368, upper bound: 7.0875827
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.23
Output dim: 6, lower bound: -7.0874811, upper bound: 7.0874811
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=7.834464073181152
rel_dist={6: [-7.106923203393791, 7.106923203247483]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1805.16 seconds
