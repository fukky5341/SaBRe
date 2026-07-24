## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 242.013661301
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635)
1: (-109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755)
2: (-144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974)
3: (-153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142)
4: (-141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818)
5: (-126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261)
6: (-121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238)
7: (-132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127)
8: (-158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799)
9: (-120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482)

## BASE Result
execution time: IAR + LP analysis = 1.08 + 8.76 = 9.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -242.0420621, upper bound: 242.0420621


# Binary Search by BASE starts (time budget: 2690.16 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=243.70181274414062
rel_dist={7: [-242.0414140657913, 242.04141406382126]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=243.70181274414062
rel_dist={7: [-242.04091628504267, 242.04091628504267]}

## Binary Search Result
Binary search time: 32.89 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2657.27 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0114063, upper bound: 242.0129199
time: 6.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0027808, upper bound: 242.0027808
time: 4.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.79 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 10.79
Output dim: 7, lower bound: -242.0114063, upper bound: 242.0129199
IS_A2, status: Status.VERIFIED, split count: 1, time: 10.79
Output dim: 7, lower bound: -242.0027808, upper bound: 242.0027808
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0153718, upper bound: 242.0173606
time: 5.93 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0028211, upper bound: 242.0028211
time: 4.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.66
Output dim: 7, lower bound: -242.0153718, upper bound: 242.0173606
IS_A2, status: Status.VERIFIED, split count: 1, time: 10.66
Output dim: 7, lower bound: -242.0028211, upper bound: 242.0028211

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -130.4297791, 103.8727341, -131.4713287, 104.7019348, -235.1317139, 235.3440552
1: -108.9829712, 92.0906067, -109.8579636, 92.8269119, -201.8098755, 201.9485779
2: -143.5179749, 93.5587921, -144.6704407, 94.3037720, -237.8217163, 238.2292328
3: -152.7482910, 81.1900024, -153.9682007, 81.8407135, -234.5890045, 235.1582031
4: -140.0591583, 107.5898438, -141.1737061, 108.4469757, -248.5061188, 248.7635498
5: -125.9344635, 98.5537338, -126.9361572, 99.3330688, -225.2675323, 225.4898987
6: -120.2822266, 115.4450302, -121.2382507, 116.3698807, -236.6520996, 236.6832886
7: -131.0889893, 110.6967621, -132.1308594, 111.5709610, -242.6599426, 242.8276215
8: -157.1396637, 107.4020386, -158.4042053, 108.2726822, -265.4122925, 265.8062439
9: -119.5088501, 117.9269180, -120.4584961, 118.8621521, -238.3710022, 238.3854065

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0030432, upper bound: 242.0032130
time: 7.37 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0107249, upper bound: 242.0119204
time: 7.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139548, upper bound: 242.0159065
time: 6.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.92
Output dim: 7, lower bound: -242.0107249, upper bound: 242.0119204
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.92
Output dim: 7, lower bound: -242.0139548, upper bound: 242.0159065

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -130.4297791, 103.8727341, -128.2993164, 102.1750793, -232.6048584, 232.1720428
1: -108.9829712, 92.0906067, -107.1907654, 90.5949478, -199.5779114, 199.2813721
2: -143.5179749, 93.5587921, -141.1857300, 92.0204620, -235.5384064, 234.7445221
3: -152.7482910, 81.1900024, -150.2456512, 79.8905334, -232.6388245, 231.4356537
4: -140.0591583, 107.5898438, -137.7583160, 105.8181229, -245.8772888, 245.3481598
5: -125.9344635, 98.5537338, -123.8605576, 96.9034882, -222.8379517, 222.4142914
6: -120.2822266, 115.4450302, -118.3207321, 113.5716400, -233.8538208, 233.7657623
7: -131.0889893, 110.6967621, -128.9285126, 108.8914948, -239.9804840, 239.6252747
8: -157.1396637, 107.4020386, -154.5771637, 105.6549454, -262.7946167, 261.9791870
9: -119.5088501, 117.9269180, -117.5448074, 115.9941330, -235.5029602, 235.4717255

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0015846, upper bound: 242.0015550
time: 6.12 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0085757, upper bound: 242.0097770
time: 6.21 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0136956, upper bound: 242.0157500
time: 5.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.87 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 27.87
Output dim: 7, lower bound: -242.0085757, upper bound: 242.0097770
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 7, lower bound: -242.0136956, upper bound: 242.0157500

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -127.9703751, 101.9098206, -128.2993164, 102.1750793, -230.1454468, 230.2091217
1: -106.9016037, 90.3441391, -107.1907654, 90.5949478, -197.4965363, 197.5348969
2: -140.7936707, 91.7790909, -141.1857300, 92.0204620, -232.8141022, 232.9648132
3: -149.8511047, 79.6702042, -150.2456512, 79.8905334, -229.7416382, 229.9158478
4: -137.3905182, 105.5337753, -137.7583160, 105.8181229, -243.2086487, 243.2920837
5: -123.5494080, 96.6556015, -123.8605576, 96.9034882, -220.4528809, 220.5161591
6: -118.0031891, 113.2591400, -118.3207321, 113.5716400, -231.5748138, 231.5798492
7: -128.5944672, 108.5971680, -128.9285126, 108.8914948, -237.4859619, 237.5256653
8: -154.1685028, 105.3619843, -154.5771637, 105.6549454, -259.8234253, 259.9391174
9: -117.2267075, 115.6770401, -117.5448074, 115.9941330, -233.2208405, 233.2218323

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0013116, upper bound: 242.0012706
time: 6.27 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9996640, upper bound: 242.0023546
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0084438, upper bound: 242.0107499
time: 6.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.04 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.04
Output dim: 7, lower bound: -241.9996640, upper bound: 242.0023546
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 27.04
Output dim: 7, lower bound: -242.0084438, upper bound: 242.0107499
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=243.70181274414062
rel_dist={7: [-242.0419761633516, 242.04197616010464]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0199215, upper bound: 242.0178282
time: 5.43 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0028466, upper bound: 242.0028466
time: 4.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.88 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.88
Output dim: 7, lower bound: -242.0199215, upper bound: 242.0178282
IS_B2, status: Status.VERIFIED, split count: 1, time: 9.88
Output dim: 7, lower bound: -242.0028466, upper bound: 242.0028466

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -131.4713287, 104.7019348, -130.4297791, 103.8727341, -235.3440552, 235.1317139
1: -109.8579636, 92.8269119, -108.9829712, 92.0906067, -201.9485779, 201.8098755
2: -144.6704407, 94.3037720, -143.5179749, 93.5587921, -238.2292328, 237.8217163
3: -153.9682007, 81.8407135, -152.7482910, 81.1900024, -235.1582031, 234.5890045
4: -141.1737061, 108.4469757, -140.0591583, 107.5898438, -248.7635498, 248.5061188
5: -126.9361572, 99.3330688, -125.9344635, 98.5537338, -225.4898987, 225.2675323
6: -121.2382507, 116.3698807, -120.2822266, 115.4450302, -236.6832886, 236.6520996
7: -132.1308594, 111.5709610, -131.0889893, 110.6967621, -242.8276215, 242.6599426
8: -158.4042053, 108.2726822, -157.1396637, 107.4020386, -265.8062439, 265.4122925
9: -120.4584961, 118.8621521, -119.5088501, 117.9269180, -238.3854065, 238.3710022

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9392945, upper bound: 241.9539806
time: 6.65 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0193176, upper bound: 242.0171316
time: 6.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.79 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 15.79
Output dim: 7, lower bound: -241.9392945, upper bound: 241.9539806
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 7, lower bound: -242.0193176, upper bound: 242.0171316

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -131.4713287, 104.7019348, -129.1262360, 102.8400269, -234.3113403, 233.8281708
1: -109.8579636, 92.8269119, -107.8883286, 91.1743698, -201.0323334, 200.7152405
2: -144.6704407, 94.3037720, -142.0838470, 92.6291504, -237.2995758, 236.3876038
3: -153.9682007, 81.8407135, -151.2268982, 80.3861237, -234.3543091, 233.0676117
4: -141.1737061, 108.4469757, -138.6604919, 106.5163803, -247.6900787, 247.1074524
5: -126.9361572, 99.3330688, -124.6800690, 97.5789566, -224.5151062, 224.0131378
6: -121.2382507, 116.3698807, -119.0823746, 114.2940674, -235.5323181, 235.4522552
7: -132.1308594, 111.5709610, -129.7851410, 109.6046066, -241.7354584, 241.3561096
8: -158.4042053, 108.2726822, -155.5646515, 106.3260727, -264.7302246, 263.8373413
9: -120.4584961, 118.8621521, -118.3248062, 116.7562408, -237.2147369, 237.1869507

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0069008, upper bound: 242.0064059
time: 5.81 seconds

## Relational analysis of IS_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144148, upper bound: 242.0129029
time: 4.99 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0178782, upper bound: 242.0157466
time: 6.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.23 seconds
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.23
Output dim: 7, lower bound: -242.0144148, upper bound: 242.0129029
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.23
Output dim: 7, lower bound: -242.0178782, upper bound: 242.0157466

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -125.3521729, 99.8408966, -129.1262360, 102.8400269, -228.1921997, 228.9671326
1: -104.7286072, 88.5295334, -107.8883286, 91.1743698, -195.9029541, 196.4178619
2: -137.9563904, 89.8996658, -142.0838470, 92.6291504, -230.5855255, 231.9835052
3: -146.8025208, 78.0892410, -151.2268982, 80.3861237, -227.1886292, 229.3161316
4: -134.5815582, 103.3770218, -138.6604919, 106.5163803, -241.0979309, 242.0375061
5: -121.0056152, 94.6529770, -124.6800690, 97.5789566, -218.5845642, 219.3330383
6: -115.6234283, 110.9901352, -119.0823746, 114.2940674, -229.9174957, 230.0725098
7: -125.9547882, 106.4100571, -129.7851410, 109.6046066, -235.5593872, 236.1951904
8: -151.0681152, 103.2325897, -155.5646515, 106.3260727, -257.3941650, 258.7972412
9: -114.8390350, 113.3405151, -118.3248062, 116.7562408, -231.5952759, 231.6652985

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0014652, upper bound: 242.0018797
time: 6.04 seconds

## Relational analysis of IS_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_B1_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0062289, upper bound: 242.0049323
time: 5.98 seconds

## Relational analysis of IS_B1_B2_A1_B2

### Relational analysis result of IS_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0088791, upper bound: 242.0072258
time: 6.15 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -128.2993164, 102.1750793, -129.1262360, 102.8400269, -231.1393280, 231.3013153
1: -107.1907654, 90.5949478, -107.8883286, 91.1743698, -198.3651428, 198.4832764
2: -141.1857300, 92.0204620, -142.0838470, 92.6291504, -233.8148651, 234.1042938
3: -150.2456512, 79.8905334, -151.2268982, 80.3861237, -230.6317444, 231.1174164
4: -137.7583160, 105.8181229, -138.6604919, 106.5163803, -244.2746735, 244.4786072
5: -123.8605576, 96.9034882, -124.6800690, 97.5789566, -221.4395142, 221.5835571
6: -118.3207321, 113.5716400, -119.0823746, 114.2940674, -232.6148071, 232.6539764
7: -128.9285126, 108.8914948, -129.7851410, 109.6046066, -238.5331116, 238.6766357
8: -154.5771637, 105.6549454, -155.5646515, 106.3260727, -260.9031372, 261.2196045
9: -117.5448074, 115.9941330, -118.3248062, 116.7562408, -234.3010559, 234.3189087

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0053456, upper bound: 242.0049227
time: 6.10 seconds

## Relational analysis of IS_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0135504, upper bound: 242.0121534
time: 7.17 seconds

## Relational analysis of IS_B1_B2_A2_B2

### Relational analysis result of IS_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0177415, upper bound: 242.0154839
time: 6.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.36 seconds
IS_B1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 24.36
Output dim: 7, lower bound: -242.0062289, upper bound: 242.0049323
IS_B1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 24.36
Output dim: 7, lower bound: -242.0088791, upper bound: 242.0072258
IS_B1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.36
Output dim: 7, lower bound: -242.0135504, upper bound: 242.0121534
IS_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.36
Output dim: 7, lower bound: -242.0177415, upper bound: 242.0154839

## BFS IS instance: IS_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -128.2993164, 102.1750793, -126.6689529, 100.8786850, -229.1779785, 228.8440247
1: -107.1907654, 90.5949478, -105.8085480, 89.4292831, -196.6200562, 196.4034729
2: -141.1857300, 92.0204620, -139.3616943, 90.8509674, -232.0366669, 231.3821259
3: -150.2456512, 79.8905334, -148.3320312, 78.8675690, -229.1132202, 228.2225647
4: -137.7583160, 105.8181229, -135.9941559, 104.4620972, -242.2204132, 241.8122864
5: -123.8605576, 96.9034882, -122.2970276, 95.6824799, -219.5430298, 219.2005005
6: -118.3207321, 113.5716400, -116.8053970, 112.1098938, -230.4306183, 230.3769684
7: -128.9285126, 108.8914948, -127.2926102, 107.5066605, -236.4351654, 236.1841125
8: -154.5771637, 105.6549454, -152.5959473, 104.2879257, -258.8650513, 258.2508850
9: -117.5448074, 115.9941330, -116.0445557, 114.5082245, -232.0530243, 232.0386658

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0050744, upper bound: 242.0046197
time: 6.11 seconds

## Relational analysis of IS_B1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_B1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0053898, upper bound: 242.0022346
time: 7.36 seconds

## Relational analysis of IS_B1_B2_A2_B2_A2

### Relational analysis result of IS_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0128609, upper bound: 242.0102457
time: 4.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.52 seconds
IS_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.52
Output dim: 7, lower bound: -242.0053898, upper bound: 242.0022346
IS_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.52
Output dim: 7, lower bound: -242.0128609, upper bound: 242.0102457
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=243.70181274414062
rel_dist={7: [-242.04203424098938, 242.04203424098944]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0189476, upper bound: 242.0210514
time: 4.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0028583, upper bound: 242.0028583
time: 4.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.62
Output dim: 7, lower bound: -242.0189476, upper bound: 242.0210514
IS_A2, status: Status.VERIFIED, split count: 1, time: 8.62
Output dim: 7, lower bound: -242.0028583, upper bound: 242.0028583

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -130.4297791, 103.8727341, -131.4713287, 104.7019348, -235.1317139, 235.3440552
1: -108.9829712, 92.0906067, -109.8579636, 92.8269119, -201.8098755, 201.9485779
2: -143.5179749, 93.5587921, -144.6704407, 94.3037720, -237.8217163, 238.2292328
3: -152.7482910, 81.1900024, -153.9682007, 81.8407135, -234.5890045, 235.1582031
4: -140.0591583, 107.5898438, -141.1737061, 108.4469757, -248.5061188, 248.7635498
5: -125.9344635, 98.5537338, -126.9361572, 99.3330688, -225.2675323, 225.4898987
6: -120.2822266, 115.4450302, -121.2382507, 116.3698807, -236.6520996, 236.6832886
7: -131.0889893, 110.6967621, -132.1308594, 111.5709610, -242.6599426, 242.8276215
8: -157.1396637, 107.4020386, -158.4042053, 108.2726822, -265.4122925, 265.8062439
9: -119.5088501, 117.9269180, -120.4584961, 118.8621521, -238.3710022, 238.3854065

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9683763, upper bound: 241.9553176
time: 5.61 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0182616, upper bound: 242.0204376
time: 5.04 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.70 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 13.70
Output dim: 7, lower bound: -241.9683763, upper bound: 241.9553176
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 13.70
Output dim: 7, lower bound: -242.0182616, upper bound: 242.0204376

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -129.1262360, 102.8400269, -131.4713287, 104.7019348, -233.8281708, 234.3113403
1: -107.8883286, 91.1743698, -109.8579636, 92.8269119, -200.7152405, 201.0323334
2: -142.0838470, 92.6291504, -144.6704407, 94.3037720, -236.3876038, 237.2995758
3: -151.2268982, 80.3861237, -153.9682007, 81.8407135, -233.0676117, 234.3543091
4: -138.6604919, 106.5163803, -141.1737061, 108.4469757, -247.1074524, 247.6900787
5: -124.6800690, 97.5789566, -126.9361572, 99.3330688, -224.0131378, 224.5151062
6: -119.0823746, 114.2940674, -121.2382507, 116.3698807, -235.4522552, 235.5323181
7: -129.7851410, 109.6046066, -132.1308594, 111.5709610, -241.3561096, 241.7354584
8: -155.5646515, 106.3260727, -158.4042053, 108.2726822, -263.8373413, 264.7302246
9: -118.3248062, 116.7562408, -120.4584961, 118.8621521, -237.1869507, 237.2147369

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0081446, upper bound: 242.0086437
time: 4.91 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0142204, upper bound: 242.0157590
time: 4.65 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0168798, upper bound: 242.0190354
time: 4.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.45 seconds
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.45
Output dim: 7, lower bound: -242.0142204, upper bound: 242.0157590
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.45
Output dim: 7, lower bound: -242.0168798, upper bound: 242.0190354

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -129.1262360, 102.8400269, -125.3521729, 99.8408966, -228.9671326, 228.1921997
1: -107.8883286, 91.1743698, -104.7286072, 88.5295334, -196.4178619, 195.9029541
2: -142.0838470, 92.6291504, -137.9563904, 89.8996658, -231.9835052, 230.5855255
3: -151.2268982, 80.3861237, -146.8025208, 78.0892410, -229.3161316, 227.1886292
4: -138.6604919, 106.5163803, -134.5815582, 103.3770218, -242.0375061, 241.0979309
5: -124.6800690, 97.5789566, -121.0056152, 94.6529770, -219.3330383, 218.5845642
6: -119.0823746, 114.2940674, -115.6234283, 110.9901352, -230.0725098, 229.9174957
7: -129.7851410, 109.6046066, -125.9547882, 106.4100571, -236.1951904, 235.5593872
8: -155.5646515, 106.3260727, -151.0681152, 103.2325897, -258.7972412, 257.3941650
9: -118.3248062, 116.7562408, -114.8390350, 113.3405151, -231.6652985, 231.5952759

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 233

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0038633, upper bound: 242.0036141
time: 6.23 seconds

## Relational analysis of IS_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0064385, upper bound: 242.0077866
time: 5.39 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0086060, upper bound: 242.0102934
time: 4.91 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -129.1262360, 102.8400269, -128.2993164, 102.1750793, -231.3013153, 231.1393280
1: -107.8883286, 91.1743698, -107.1907654, 90.5949478, -198.4832764, 198.3651428
2: -142.0838470, 92.6291504, -141.1857300, 92.0204620, -234.1042938, 233.8148651
3: -151.2268982, 80.3861237, -150.2456512, 79.8905334, -231.1174164, 230.6317444
4: -138.6604919, 106.5163803, -137.7583160, 105.8181229, -244.4786072, 244.2746735
5: -124.6800690, 97.5789566, -123.8605576, 96.9034882, -221.5835571, 221.4395142
6: -119.0823746, 114.2940674, -118.3207321, 113.5716400, -232.6539764, 232.6148071
7: -129.7851410, 109.6046066, -128.9285126, 108.8914948, -238.6766357, 238.5331116
8: -155.5646515, 106.3260727, -154.5771637, 105.6549454, -261.2196045, 260.9031372
9: -118.3248062, 116.7562408, -117.5448074, 115.9941330, -234.3189087, 234.3010559

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0066777, upper bound: 242.0071205
time: 5.70 seconds

## Relational analysis of IS_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0139565, upper bound: 242.0154072
time: 4.75 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0166152, upper bound: 242.0188989
time: 5.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.95 seconds
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 20.95
Output dim: 7, lower bound: -242.0064385, upper bound: 242.0077866
IS_A1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 20.95
Output dim: 7, lower bound: -242.0086060, upper bound: 242.0102934
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 20.95
Output dim: 7, lower bound: -242.0139565, upper bound: 242.0154072
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.95
Output dim: 7, lower bound: -242.0166152, upper bound: 242.0188989

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -124.3062134, 98.9952240, -128.2993164, 102.1750793, -226.4812927, 227.2945251
1: -103.7921066, 87.7248611, -107.1907654, 90.5949478, -194.3870544, 194.9156189
2: -136.7356567, 89.1137314, -141.1857300, 92.0204620, -228.7561035, 230.2994690
3: -145.5236511, 77.3834839, -150.2456512, 79.8905334, -225.4141846, 227.6291046
4: -133.4133759, 102.4507904, -137.7583160, 105.8181229, -239.2314911, 240.2091064
5: -119.9855728, 93.7984772, -123.8605576, 96.9034882, -216.8890381, 217.6590271
6: -114.6034088, 110.0018997, -118.3207321, 113.5716400, -228.1750336, 228.3226166
7: -124.8618317, 105.4574585, -128.9285126, 108.8914948, -233.7533264, 234.3859406
8: -149.7491608, 102.3247757, -154.5771637, 105.6549454, -255.4040833, 256.9018860
9: -113.7944641, 112.3057785, -117.5448074, 115.9941330, -229.7886047, 229.8505707

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9990999, upper bound: 241.9970303
time: 5.19 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0008813, upper bound: 242.0032444
time: 5.18 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0085035, upper bound: 242.0103596
time: 6.59 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -126.6689529, 100.8786850, -128.2993164, 102.1750793, -228.8440247, 229.1779785
1: -105.8085480, 89.4292831, -107.1907654, 90.5949478, -196.4034729, 196.6200562
2: -139.3616943, 90.8509674, -141.1857300, 92.0204620, -231.3821259, 232.0366669
3: -148.3320312, 78.8675690, -150.2456512, 79.8905334, -228.2225647, 229.1132202
4: -135.9941559, 104.4620972, -137.7583160, 105.8181229, -241.8122864, 242.2204132
5: -122.2970276, 95.6824799, -123.8605576, 96.9034882, -219.2005005, 219.5430298
6: -116.8053970, 112.1098938, -118.3207321, 113.5716400, -230.3769684, 230.4306183
7: -127.2926102, 107.5066605, -128.9285126, 108.8914948, -236.1841125, 236.4351654
8: -152.5959473, 104.2879257, -154.5771637, 105.6549454, -258.2508850, 258.8650208
9: -116.0445557, 114.5082245, -117.5448074, 115.9941330, -232.0386658, 232.0530243

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0063709, upper bound: 242.0069251
time: 4.90 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0038455, upper bound: 242.0071843
time: 4.88 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0114265, upper bound: 242.0141110
time: 4.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.56 seconds
IS_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 19.56
Output dim: 7, lower bound: -242.0008813, upper bound: 242.0032444
IS_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 19.56
Output dim: 7, lower bound: -242.0085035, upper bound: 242.0103596
IS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 19.56
Output dim: 7, lower bound: -242.0038455, upper bound: 242.0071843
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 19.56
Output dim: 7, lower bound: -242.0114265, upper bound: 242.0141110

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -126.6689529, 100.8786850, -118.7319183, 94.6117325, -221.2806702, 219.6105804
1: -105.8085480, 89.4292831, -99.1450424, 83.8185349, -189.6270599, 188.5743103
2: -139.3616943, 90.8509674, -130.6122437, 85.1732254, -224.5348969, 221.4631958
3: -148.3320312, 78.8675690, -139.0087280, 73.8999252, -222.2319641, 217.8762970
4: -135.9941559, 104.4620972, -127.4941330, 97.9313126, -233.9254761, 231.9562378
5: -122.2970276, 95.6824799, -114.6694183, 89.7484436, -212.0454712, 210.3518982
6: -116.8053970, 112.1098938, -109.5087204, 105.0966034, -221.9019775, 221.6186066
7: -127.2926102, 107.5066605, -119.3291931, 100.8465652, -228.1391296, 226.8358154
8: -152.5959473, 104.2879257, -143.0311127, 97.7613525, -250.3572693, 247.3190308
9: -116.0445557, 114.5082245, -108.8197937, 107.3722534, -223.4167786, 223.3280182

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0005031, upper bound: 242.0009998
time: 4.80 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0114265, upper bound: 242.0141111
time: 6.39 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0114265, upper bound: 242.0141094
time: 4.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.24 seconds
IS_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 21.24
Output dim: 7, lower bound: -242.0114265, upper bound: 242.0141111
IS_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 21.24
Output dim: 7, lower bound: -242.0114265, upper bound: 242.0141094

## BFS IS instance: IS_A1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -120.7468262, 96.1833649, -118.7319183, 94.6117325, -215.3585358, 214.9152679
1: -100.8796616, 85.2856522, -99.1450424, 83.8185349, -184.6981812, 184.4306793
2: -132.8639984, 86.6499405, -130.6122437, 85.1732254, -218.0372162, 217.2621765
3: -141.3900757, 75.2038422, -139.0087280, 73.8999252, -215.2900085, 214.2125397
4: -129.6575623, 99.6231003, -127.4941330, 97.9313126, -227.5888672, 227.1172333
5: -116.5745697, 91.2481689, -114.6694183, 89.7484436, -206.3230133, 205.9175873
6: -111.3917999, 106.8991318, -109.5087204, 105.0966034, -216.4883881, 216.4078217
7: -121.3960342, 102.5433807, -119.3291931, 100.8465652, -222.2425842, 221.8725281
8: -145.4971008, 99.4137115, -143.0311127, 97.7613525, -243.2584229, 242.4448090
9: -110.6676102, 109.2461014, -108.8197937, 107.3722534, -218.0398560, 218.0658417

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0004835, upper bound: 242.0009998
time: 4.81 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0060401, upper bound: 242.0072249
time: 5.90 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0085741, upper bound: 242.0118585
time: 5.22 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -125.4435806, 99.9147873, -118.7319183, 94.6117325, -220.0552979, 218.6466827
1: -104.7889175, 88.5740967, -99.1450424, 83.8185349, -188.6074524, 187.7191467
2: -138.0210724, 89.9857635, -130.6122437, 85.1732254, -223.1942902, 220.5979767
3: -146.8959808, 78.1156693, -139.0087280, 73.8999252, -220.7958984, 217.1243896
4: -134.6730957, 103.4589844, -127.4941330, 97.9313126, -232.6044006, 230.9531250
5: -121.1208725, 94.7733612, -114.6694183, 89.7484436, -210.8693237, 209.4427795
6: -115.6791840, 111.0323792, -109.5087204, 105.0966034, -220.7757874, 220.5411072
7: -126.0654297, 106.4830246, -119.3291931, 100.8465652, -226.9119720, 225.8122101
8: -151.1301727, 103.2894669, -143.0311127, 97.7613525, -248.8914795, 246.3205719
9: -114.9279099, 113.4208603, -108.8197937, 107.3722534, -222.3001404, 222.2406311

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0005031, upper bound: 242.0009998
time: 5.38 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0060401, upper bound: 242.0072242
time: 5.16 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0085741, upper bound: 242.0118393
time: 4.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 20.40 seconds
IS_A1_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 20.40
Output dim: 7, lower bound: -242.0060401, upper bound: 242.0072249
IS_A1_A2_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 20.40
Output dim: 7, lower bound: -242.0085741, upper bound: 242.0118585
IS_A1_A2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 20.40
Output dim: 7, lower bound: -242.0060401, upper bound: 242.0072242
IS_A1_A2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 20.40
Output dim: 7, lower bound: -242.0085741, upper bound: 242.0118393
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=243.70181274414062
rel_dist={7: [-242.04206207992112, 242.04206207992115]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 457.14 seconds
