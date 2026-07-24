## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 143.61867486269998
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766)
1: (-62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901)
2: (-83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913)
3: (-88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352)
4: (-81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725)
5: (-73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507)
6: (-70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321)
7: (-76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043)
8: (-92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513)
9: (-69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561)

## BASE Result
execution time: IAR + LP analysis = 1.45 + 8.25 = 9.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624614, upper bound: 143.7624614


# Binary Search by BASE starts (time budget: 1990.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search Result
Binary search time: 33.31 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1956.99 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7329251, upper bound: 143.7351171
time: 6.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7622839, upper bound: 143.7622839
time: 6.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.91
Output dim: 4, lower bound: -143.7329251, upper bound: 143.7351171
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.91
Output dim: 4, lower bound: -143.7622839, upper bound: 143.7622839

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -75.5166473, 60.2843056, -75.4620590, 60.2116890, -135.7283020, 135.7463684
1: -62.4711227, 53.3862686, -62.5184593, 53.3833237, -115.8544464, 115.9047241
2: -82.8460770, 53.9459114, -82.7861176, 53.9993553, -136.8454285, 136.7320099
3: -88.1652603, 46.6531601, -88.0941849, 46.5770302, -134.7422638, 134.7473145
4: -81.4115829, 62.4901505, -81.3885803, 62.5474739, -143.9590607, 143.8787231
5: -72.6913757, 56.5188599, -72.6200180, 56.5331421, -129.2245178, 129.1388855
6: -69.7521362, 66.5912628, -69.7578125, 66.5460052, -136.2981415, 136.3490753
7: -75.4242477, 63.4810486, -75.4427719, 63.5288620, -138.9530945, 138.9238281
8: -91.4648590, 63.3575401, -91.3506012, 63.3049164, -154.7697754, 154.7081451
9: -68.7352753, 67.5297241, -68.8379593, 67.6329422, -136.3682251, 136.3676758

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7036484, upper bound: 143.7080901
time: 7.03 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6981089, upper bound: 143.7011070
time: 6.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -75.4379196, 60.1935806, -76.0228043, 60.6544914, -136.0923920, 136.2163696
1: -62.4942665, 53.3639336, -62.9943390, 53.7792549, -116.2735214, 116.3582764
2: -82.7585068, 53.9781189, -83.4031906, 54.4029045, -137.1614075, 137.3813171
3: -88.0636444, 46.5641823, -88.7539597, 46.9208794, -134.9845123, 135.3181458
4: -81.3618927, 62.5229797, -81.9910431, 63.0132256, -144.3751068, 144.5140228
5: -72.5973358, 56.5129662, -73.1580811, 56.9551659, -129.5524750, 129.6710510
6: -69.7321014, 66.5246277, -70.2768860, 67.0410385, -136.7731323, 136.8015137
7: -75.4163895, 63.5050621, -76.0090866, 64.0053253, -139.4217072, 139.5141296
8: -91.3206482, 63.2856140, -92.0233536, 63.7623940, -155.0830383, 155.3089447
9: -68.8098526, 67.6032944, -69.3594666, 68.1468353, -136.9566498, 136.9627380

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6720134, upper bound: 143.6821593
time: 6.07 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6417950, upper bound: 143.6417950
time: 5.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 12.68
Output dim: 4, lower bound: -143.7036484, upper bound: 143.7080901
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.68
Output dim: 4, lower bound: -143.6981089, upper bound: 143.7011070
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.68
Output dim: 4, lower bound: -143.6720134, upper bound: 143.6821593
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.68
Output dim: 4, lower bound: -143.6417950, upper bound: 143.6417950

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -75.5166473, 60.2843056, -66.3601685, 53.0202103, -128.5368195, 126.6444626
1: -62.4711227, 53.3862686, -54.7785530, 46.9473763, -109.4185028, 108.1648254
2: -82.8460770, 53.9459114, -72.6824112, 47.5009346, -130.3470154, 126.6283188
3: -88.1652603, 46.6531601, -77.4608078, 40.9182472, -129.0834808, 124.1139374
4: -81.4115829, 62.4901505, -71.6432800, 55.0747643, -136.4863434, 134.1334229
5: -72.6913757, 56.5188599, -63.8880577, 49.7958908, -122.4872665, 120.4069214
6: -69.7521362, 66.5912628, -61.3425369, 58.4863472, -128.2384796, 127.9337997
7: -75.4242477, 63.4810486, -66.2944336, 55.8586769, -131.2828979, 129.7754822
8: -91.4648590, 63.3575401, -80.3490677, 55.7951088, -147.2599640, 143.7066040
9: -68.7352753, 67.5297241, -60.5214767, 59.4044266, -128.1397095, 128.0511780

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6696115, upper bound: 143.6763559
time: 7.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6957725, upper bound: 143.7004320
time: 7.82 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -71.3076019, 56.9580956, -58.2362404, 46.5325279, -117.8401260, 115.1943359
1: -58.8913956, 50.4074097, -47.8103752, 41.1346283, -100.0260162, 98.2177658
2: -78.1773834, 50.9440727, -63.5800972, 41.6293449, -119.8067322, 114.5241699
3: -83.2509613, 44.0398102, -67.9612579, 35.8000412, -119.0509949, 112.0010681
4: -76.8992081, 59.0329018, -63.0084496, 48.3861809, -125.2853775, 122.0413513
5: -68.6536026, 53.4013596, -56.0820618, 43.7602158, -112.4138184, 109.4834213
6: -65.8631058, 62.8618851, -53.8069382, 51.3157349, -117.1788406, 116.6688232
7: -71.1987381, 59.9359207, -58.0897484, 48.9592781, -120.1580200, 118.0256653
8: -86.3777847, 59.8761787, -70.4995804, 49.0052147, -135.3829956, 130.3757629
9: -64.8900528, 63.7246666, -53.1417122, 51.9977188, -116.8877716, 116.8663788

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6637816, upper bound: 143.6691806
time: 6.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6902026, upper bound: 143.6933680
time: 7.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -75.4379196, 60.1935806, -75.7754288, 60.4595566, -135.8974609, 135.9689941
1: -62.4942665, 53.3639336, -62.7865219, 53.6056404, -116.0999069, 116.1504517
2: -82.7585068, 53.9781189, -83.1312408, 54.2274513, -136.9859619, 137.1093292
3: -88.0636444, 46.5641823, -88.4644012, 46.7679405, -134.8315582, 135.0285797
4: -81.3618927, 62.5229797, -81.7224503, 62.8103371, -144.1722260, 144.2453918
5: -72.5973358, 56.5129662, -72.9203491, 56.7705574, -129.3678436, 129.4333191
6: -69.7321014, 66.5246277, -70.0489120, 66.8218460, -136.5539246, 136.5735474
7: -75.4163895, 63.5050621, -75.7611084, 63.7964745, -139.2128601, 139.2661438
8: -91.3206482, 63.2856140, -91.7253265, 63.5594482, -154.8800964, 155.0109406
9: -68.8098526, 67.6032944, -69.1330185, 67.9258499, -136.7356720, 136.7363129

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6263989, upper bound: 143.6342751
time: 6.77 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6457118, upper bound: 143.6529004
time: 8.67 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6670033, upper bound: 143.6769562
time: 6.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -75.1824570, 59.9929771, -97.5423889, 77.7365036, -152.9189606, 157.5353546
1: -62.2790833, 53.1843796, -81.1014328, 68.8907394, -131.1698303, 134.2858124
2: -82.4772949, 53.7965660, -106.9554520, 69.4951782, -151.9724579, 160.7520142
3: -87.7640610, 46.4060669, -113.8497009, 60.1707802, -147.9348450, 160.2557526
4: -81.0845184, 62.3132820, -105.0295410, 80.6163635, -161.7008820, 167.3428192
5: -72.3520889, 56.3226204, -93.9048538, 72.9319458, -145.2840118, 150.2274780
6: -69.4965515, 66.2977982, -89.9386978, 86.1439285, -155.6404724, 156.2364960
7: -75.1597900, 63.2891960, -97.3581924, 82.0536880, -157.2134705, 160.6473846
8: -91.0122375, 63.0760803, -117.8584747, 81.4248352, -172.4370270, 180.9345551
9: -68.5754013, 67.3753052, -88.8761597, 87.3950424, -155.9704437, 156.2514496

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6114318, upper bound: 143.6089765
time: 7.02 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6360245, upper bound: 143.6360245
time: 5.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.23 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6696115, upper bound: 143.6763559
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6957725, upper bound: 143.7004320
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6637816, upper bound: 143.6691806
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6902026, upper bound: 143.6933680
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6457118, upper bound: 143.6529004
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6670033, upper bound: 143.6769562
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6114318, upper bound: 143.6089765
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 18.23
Output dim: 4, lower bound: -143.6360245, upper bound: 143.6360245

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -54.7785797, 43.8699646, -63.0944099, 50.4335823, -105.2121582, 106.9643707
1: -44.9663086, 38.7933960, -52.0136757, 44.6446686, -89.6109695, 90.8070602
2: -59.8750191, 39.2165909, -69.0709763, 45.1783104, -105.0533218, 108.2875595
3: -63.9433289, 33.8884277, -73.6380463, 38.8994255, -102.8427582, 107.5264664
4: -59.1821442, 45.3875084, -68.1411514, 52.3818283, -111.5639496, 113.5286560
5: -52.8066025, 41.0475960, -60.7498093, 47.3567581, -100.1633606, 101.7973938
6: -50.6971512, 48.2587852, -58.3386154, 55.5962448, -106.2933960, 106.5973892
7: -54.5866928, 45.9521370, -63.0181580, 53.0957756, -107.6824646, 108.9702835
8: -66.5126190, 46.2618904, -76.4242249, 53.1069374, -119.6195526, 122.6861115
9: -49.6636887, 48.6880112, -57.5194740, 56.4374237, -106.1011124, 106.2074890

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6371803, upper bound: 143.6406888
time: 7.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6645189, upper bound: 143.6712414
time: 6.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -69.4262695, 55.4635162, -66.3601685, 53.0202103, -122.4464798, 121.8236847
1: -57.2995262, 49.0872192, -54.7785530, 46.9473763, -104.2469025, 103.8657684
2: -76.1045380, 49.6180687, -72.6824112, 47.5009346, -123.6054688, 122.3004761
3: -81.0419769, 42.8915062, -77.4608078, 40.9182472, -121.9602051, 120.3523102
4: -74.8741302, 57.4800186, -71.6432800, 55.0747643, -129.9488983, 129.1232910
5: -66.8403397, 51.9837456, -63.8880577, 49.7958908, -116.6362305, 115.8718033
6: -64.1379623, 61.1916962, -61.3425369, 58.4863472, -122.6243057, 122.5342178
7: -69.3232117, 58.3398323, -66.2944336, 55.8586769, -125.1818848, 124.6342621
8: -84.1293106, 58.3443871, -80.3490677, 55.7951088, -139.9244232, 138.6934509
9: -63.1461601, 62.0107574, -60.5214767, 59.4044266, -122.5505829, 122.5322342

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6077515, upper bound: 143.6175428
time: 8.15 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6916747, upper bound: 143.6962933
time: 6.81 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -50.8384933, 40.7544289, -55.0955467, 44.0431633, -94.8816528, 95.8499756
1: -41.6613503, 36.0165215, -45.1735497, 38.9242554, -80.5856018, 81.1900711
2: -55.4929581, 36.4211044, -60.1015320, 39.3983383, -94.8912964, 96.5226288
3: -59.3439102, 31.4592209, -64.2922745, 33.8619843, -93.2058868, 95.7514954
4: -54.9702911, 42.1432838, -59.6420937, 45.7983398, -100.7686310, 101.7853775
5: -49.0312996, 38.1214943, -53.0656242, 41.4162674, -90.4475555, 91.1871109
6: -47.0767517, 44.7878876, -50.9184380, 48.5464478, -95.6231918, 95.7063293
7: -50.6265106, 42.6333542, -54.9375114, 46.3085213, -96.9350281, 97.5708618
8: -61.7399216, 42.9921570, -66.7234802, 46.4163437, -108.1562653, 109.7156296
9: -46.0827370, 45.1433563, -50.2620888, 49.1584930, -95.2412262, 95.4054413

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6277973, upper bound: 143.6298161
time: 7.01 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6585921, upper bound: 143.6639114
time: 6.39 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -65.2323685, 52.1461868, -58.2362404, 46.5325279, -111.7648926, 110.3824310
1: -53.7361908, 46.1214523, -47.8103752, 41.1346283, -94.8708115, 93.9318085
2: -71.4496689, 46.6271935, -63.5800972, 41.6293449, -113.0790100, 110.2072906
3: -76.1405869, 40.2902222, -67.9612579, 35.8000412, -111.9406128, 108.2514801
4: -70.3764191, 54.0346451, -63.0084496, 48.3861809, -118.7625885, 117.0430908
5: -62.8161812, 48.8740120, -56.0820618, 43.7602158, -106.5763931, 104.9560699
6: -60.2653847, 57.4767189, -53.8069382, 51.3157349, -111.5811157, 111.2836609
7: -65.1092758, 54.8044205, -58.0897484, 48.9592781, -114.0685577, 112.8941650
8: -79.0614166, 54.8755760, -70.4995804, 49.0052147, -128.0666199, 125.3751526
9: -59.3125648, 58.2179642, -53.1417122, 51.9977188, -111.3102875, 111.3596802

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6016509, upper bound: 143.6103391
time: 7.43 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6859163, upper bound: 143.6890688
time: 6.39 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -71.9049606, 57.3632622, -54.7902946, 43.5995407, -115.5044861, 112.1535568
1: -59.4854927, 50.8545532, -44.9289703, 38.7098618, -98.1953583, 95.7835236
2: -78.8313904, 51.4557076, -59.7744675, 39.2263947, -118.0577850, 111.2301788
3: -83.9375305, 44.3782310, -63.9697456, 33.7859230, -117.7234497, 108.3479538
4: -77.5325317, 59.5875320, -58.9870834, 45.3164444, -122.8489761, 118.5745926
5: -69.1980972, 53.8684998, -52.7296028, 41.0371628, -110.2352448, 106.5980988
6: -66.4671097, 63.3742752, -50.6629143, 48.1181526, -114.5852661, 114.0371857
7: -71.8661880, 60.5236549, -54.6724396, 46.0678635, -117.9340363, 115.1960907
8: -87.0051041, 60.2839203, -66.0516815, 45.6281471, -132.6332397, 126.3356018
9: -65.5539169, 64.3701782, -49.7726402, 48.6617470, -114.2156601, 114.1428223

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5901807, upper bound: 143.5955875
time: 8.53 seconds

## Relational analysis of IS_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6193607, upper bound: 143.6270796
time: 7.42 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6404574, upper bound: 143.6477783
time: 7.54 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -75.4379196, 60.1935806, -70.9414368, 56.5996780, -132.0375671, 131.1350098
1: -62.4942665, 53.3639336, -58.6741219, 50.1796494, -112.6739197, 112.0380478
2: -82.7585068, 53.9781189, -77.7669907, 50.7811356, -133.5396423, 131.7451172
3: -88.0636444, 46.5641823, -82.8174133, 43.7814598, -131.8450928, 129.3815918
4: -81.3618927, 62.5229797, -76.4946289, 58.8091278, -140.1710205, 139.0175934
5: -72.5973358, 56.5129662, -68.2696915, 53.1607132, -125.7580414, 124.7826538
6: -69.7321014, 66.5246277, -65.5838089, 62.5160980, -132.2481995, 132.1084290
7: -75.4163895, 63.5050621, -70.9044418, 59.7167511, -135.1331482, 134.4094696
8: -91.3206482, 63.2856140, -85.8449326, 59.4922028, -150.8128510, 149.1305389
9: -68.8098526, 67.6032944, -64.6815567, 63.5168762, -132.3267212, 132.2848358

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6204929, upper bound: 143.6283314
time: 6.54 seconds

## Relational analysis of IS_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6439002, upper bound: 143.6552596
time: 7.79 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6625824, upper bound: 143.6726270
time: 8.48 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -75.1824570, 59.9929771, -92.7847443, 73.9378281, -149.1202850, 152.7777252
1: -62.2790833, 53.1843796, -77.0551910, 65.5191956, -127.7982788, 130.2395630
2: -82.4772949, 53.7965660, -101.6757355, 66.1023636, -148.5796509, 155.4723053
3: -87.7640610, 46.4060669, -108.2916336, 57.2319260, -144.9959869, 154.6976929
4: -81.0845184, 62.3132820, -99.8842316, 76.6764984, -157.7610168, 162.1974945
5: -72.3520889, 56.3226204, -89.3289337, 69.3778458, -141.7299042, 145.6515350
6: -69.4965515, 66.2977982, -85.5430298, 81.9075851, -151.4041443, 151.8408203
7: -75.1597900, 63.2891960, -92.5772018, 78.0383911, -153.1981659, 155.8663940
8: -91.0122375, 63.0760803, -112.0712662, 77.4210510, -168.4332581, 175.1473389
9: -68.5754013, 67.3753052, -84.4948578, 83.0564651, -151.6318665, 151.8701630

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6134296, upper bound: 143.6113673
time: 5.90 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6304012, upper bound: 143.6304012
time: 4.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.48 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6371803, upper bound: 143.6406888
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6645189, upper bound: 143.6712414
IS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6077515, upper bound: 143.6175428
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6916747, upper bound: 143.6962933
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6277973, upper bound: 143.6298161
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6585921, upper bound: 143.6639114
IS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6016509, upper bound: 143.6103391
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6859163, upper bound: 143.6890688
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6193607, upper bound: 143.6270796
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6404574, upper bound: 143.6477783
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6439002, upper bound: 143.6552596
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6625824, upper bound: 143.6726270
IS_A2_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6134296, upper bound: 143.6113673
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.48
Output dim: 4, lower bound: -143.6304012, upper bound: 143.6304012

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -51.4968567, 41.2383499, -42.7955132, 34.0840607, -85.5809174, 84.0338593
1: -42.2049370, 36.4734917, -34.8832092, 30.2591209, -72.4640579, 71.3566971
2: -56.2180557, 36.8876076, -46.4402580, 30.7051964, -86.9232330, 83.3278580
3: -60.1089211, 31.8732471, -49.9282417, 26.3806305, -86.4895477, 81.8014679
4: -55.6298637, 42.6557999, -46.1880188, 35.4511757, -91.0810394, 88.8438110
5: -49.6524086, 38.5827827, -41.2385254, 32.1074257, -81.7598267, 79.8213043
6: -47.6834068, 45.3453445, -39.6379890, 37.5598183, -85.2432251, 84.9833221
7: -51.2852020, 43.1830406, -42.6137772, 35.9263153, -87.2115173, 85.7968140
8: -62.4965973, 43.4623070, -51.5418663, 35.7494888, -98.2460709, 95.0041580
9: -46.6520271, 45.6930695, -38.8223763, 37.8664818, -84.5185013, 84.5154419

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5908589, upper bound: 143.5968149
time: 7.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6311349, upper bound: 143.6348961
time: 7.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -54.7785797, 43.8699646, -58.3848076, 46.6645508, -101.4431152, 102.2547760
1: -44.9663086, 38.7933960, -48.0253754, 41.3133926, -86.2796936, 86.8187714
2: -59.8750191, 39.2165909, -63.8460426, 41.8243103, -101.6993256, 103.0626144
3: -63.9433289, 33.8884277, -68.1273193, 35.9921494, -99.9354782, 102.0157471
4: -59.1821442, 45.3875084, -63.0519295, 48.4748955, -107.6570282, 108.4394379
5: -52.8066025, 41.0475960, -56.2155914, 43.8331528, -96.6397552, 97.2631836
6: -50.6971512, 48.2587852, -53.9951363, 51.4137535, -102.1109009, 102.2539215
7: -54.5866928, 45.9521370, -58.2835350, 49.1160545, -103.7027359, 104.2356567
8: -66.5126190, 46.2618904, -70.6978760, 49.1369705, -115.6495819, 116.9597626
9: -49.6636887, 48.6880112, -53.1852951, 52.1379814, -101.8016510, 101.8733063

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6538692, upper bound: 143.6615222
time: 6.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6545721, upper bound: 143.6623154
time: 7.40 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -64.6687546, 51.6594658, -66.3601685, 53.0202103, -117.6889648, 118.0196228
1: -53.2548866, 45.7179985, -54.7785530, 46.9473763, -100.2022629, 100.4965515
2: -70.8189468, 46.2263451, -72.6824112, 47.5009346, -118.3198853, 118.9087524
3: -75.4772720, 39.9544106, -77.4608078, 40.9182472, -116.3955002, 117.4152069
4: -69.7253494, 53.5392532, -71.6432800, 55.0747643, -124.8001099, 125.1825333
5: -62.2608528, 48.4255257, -63.8880577, 49.7958908, -112.0567474, 112.3135834
6: -59.7450142, 56.9539146, -61.3425369, 58.4863472, -118.2313538, 118.2964478
7: -64.5383301, 54.3211327, -66.2944336, 55.8586769, -120.3970032, 120.6155624
8: -78.3407059, 54.3388481, -80.3490677, 55.7951088, -134.1357880, 134.6879120
9: -58.7623749, 57.6683769, -60.5214767, 59.4044266, -118.1668015, 118.1898422

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6630762, upper bound: 143.6643727
time: 7.59 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6869825, upper bound: 143.6919673
time: 7.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -47.6013641, 38.1552734, -36.6797867, 29.1797562, -76.7811050, 74.8350525
1: -38.9360046, 33.7242851, -29.7186394, 25.9082127, -64.8442154, 63.4429169
2: -51.8823738, 34.1200294, -39.5277977, 26.3295078, -78.2118835, 73.6478271
3: -55.5624733, 29.4674892, -42.7361069, 22.5818119, -78.1442871, 72.2035904
4: -51.4686317, 39.4453087, -39.7691231, 30.3946877, -81.8633194, 79.2144318
5: -45.9183922, 35.6878853, -35.3562965, 27.5574017, -73.4757919, 71.0441742
6: -44.1015511, 41.9112015, -34.0186157, 32.1795692, -76.2811203, 75.9298172
7: -47.3696365, 39.8991890, -36.4672623, 30.6886559, -78.0582886, 76.3664398
8: -57.7713509, 40.2357979, -44.0614929, 30.6695080, -88.4408493, 84.2972717
9: -43.1162834, 42.1932144, -33.3162384, 32.3513527, -75.4676361, 75.5094528

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5810630, upper bound: 143.5860475
time: 6.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6215539, upper bound: 143.6238746
time: 7.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -50.8384933, 40.7544289, -50.5055275, 40.3639107, -91.2024078, 91.2599564
1: -41.6613503, 36.0165215, -41.3063965, 35.6768799, -77.3382263, 77.3229218
2: -55.4929581, 36.4211044, -54.9895821, 36.1305428, -91.6235046, 91.4106903
3: -59.3439102, 31.4592209, -58.9307899, 31.0394802, -90.3833694, 90.3899994
4: -54.9702911, 42.1432838, -54.6816750, 41.9865036, -96.9567947, 96.8249588
5: -49.0312996, 38.1214943, -48.6533852, 37.9761925, -87.0074692, 86.7748795
6: -47.0767517, 44.7878876, -46.6956673, 44.4743690, -91.5511169, 91.4835434
7: -50.6265106, 42.6333542, -50.3229637, 42.4292793, -93.0557861, 92.9563065
8: -61.7399216, 42.9921570, -61.1270447, 42.5356865, -104.2756042, 104.1191864
9: -46.0827370, 45.1433563, -46.0482140, 44.9811783, -91.0639191, 91.1915741

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6303346, upper bound: 143.6330331
time: 7.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6530526, upper bound: 143.6587880
time: 6.24 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -60.5048294, 48.3615265, -58.2362404, 46.5325279, -107.0373535, 106.5977631
1: -49.7366447, 42.7799263, -47.8103752, 41.1346283, -90.8712692, 90.5902710
2: -66.2019806, 43.2634850, -63.5800972, 41.6293449, -107.8313293, 106.8435822
3: -70.6114120, 37.3736610, -67.9612579, 35.8000412, -106.4114456, 105.3349075
4: -65.2669907, 50.1145096, -63.0084496, 48.3861809, -113.6531677, 113.1229553
5: -58.2671623, 45.3368492, -56.0820618, 43.7602158, -102.0273743, 101.4189072
6: -55.9066811, 53.2799225, -53.8069382, 51.3157349, -107.2224121, 107.0868607
7: -60.3552971, 50.8142738, -58.0897484, 48.9592781, -109.3145752, 108.9040222
8: -73.3142166, 50.8919945, -70.4995804, 49.0052147, -122.3194275, 121.3915710
9: -54.9631500, 53.9081535, -53.1417122, 51.9977188, -106.9608688, 107.0498657

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6428470, upper bound: 143.6490374
time: 6.84 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809016, upper bound: 143.6843873
time: 5.75 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -58.4692192, 46.6381187, -52.4196243, 41.7031784, -100.1723709, 99.0577393
1: -47.9180489, 41.2911415, -42.9086075, 37.0248222, -84.9428711, 84.1997452
2: -63.9617920, 41.9416313, -57.1479492, 37.5513458, -101.5131302, 99.0895767
3: -68.3604660, 36.0064278, -61.2082176, 32.3184395, -100.6788940, 97.2146454
4: -63.0415688, 48.3461189, -56.4385643, 43.3455238, -106.3870926, 104.7846832
5: -56.3209839, 43.8610802, -50.4591103, 39.2713928, -95.5923767, 94.3201904
6: -54.0100212, 51.3702660, -48.4705887, 46.0108910, -100.0209122, 99.8408508
7: -58.3884392, 49.2146950, -52.2934875, 44.0731812, -102.4616241, 101.5081787
8: -70.5515594, 48.7104797, -63.1472626, 43.6025581, -114.1541061, 111.8577423
9: -53.2001266, 52.0518150, -47.5982246, 46.5107231, -99.7108383, 99.6500244

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5614357, upper bound: 143.5671249
time: 6.54 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6179748, upper bound: 143.6262535
time: 7.44 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6158864, upper bound: 143.6245780
time: 7.39 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -67.8946609, 54.1746788, -54.7902946, 43.5995407, -111.4941940, 108.9649734
1: -56.0544853, 48.0117989, -44.9289703, 38.7098618, -94.7643433, 92.9407578
2: -74.3995819, 48.6226654, -59.7744675, 39.2263947, -113.6259766, 108.3971329
3: -79.2782440, 41.8868713, -63.9697456, 33.7859230, -113.0641479, 105.8566055
4: -73.2084579, 56.2620163, -58.9870834, 45.3164444, -118.5249023, 115.2490845
5: -65.3534164, 50.8859940, -52.7296028, 41.0371628, -106.3905716, 103.6155930
6: -62.7548218, 59.7994766, -50.6629143, 48.1181526, -110.8729706, 110.4623642
7: -67.8489380, 57.1547356, -54.6724396, 46.0678635, -113.9167938, 111.8271790
8: -82.1141891, 56.8811722, -66.0516815, 45.6281471, -127.7423401, 122.9328537
9: -61.8756218, 60.7218056, -49.7726402, 48.6617470, -110.5373535, 110.4944458

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5835427, upper bound: 143.5890609
time: 7.94 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5803175, upper bound: 143.5873316
time: 10.19 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6388698, upper bound: 143.6464926
time: 7.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6383727, upper bound: 143.6461264
time: 6.97 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -61.9123039, 49.4032249, -68.4462433, 54.6117439, -116.5240479, 117.8494492
1: -50.8366470, 43.7299919, -56.5333405, 48.4073792, -99.2440262, 100.2633209
2: -67.7896957, 44.3958549, -75.0069199, 49.0136452, -116.8033447, 119.4027634
3: -72.3853836, 38.1334724, -79.9182434, 42.2312431, -114.6165924, 118.0517120
4: -66.7714386, 51.2123489, -73.8010330, 56.7351761, -123.5066147, 125.0133820
5: -59.6338692, 46.4421997, -65.8749390, 51.3037758, -110.9376450, 112.3171387
6: -57.1835289, 54.4350014, -63.2707329, 60.2888412, -117.4723663, 117.7057343
7: -61.8501205, 52.1198120, -68.4016418, 57.6184502, -119.4685593, 120.5214539
8: -74.7541428, 51.6369133, -82.7948608, 57.3636055, -132.1177368, 134.4317627
9: -56.3691292, 55.2032089, -62.3911514, 61.2419205, -117.6110535, 117.5943451

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6031199, upper bound: 143.6101436
time: 9.43 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6378572, upper bound: 143.6493985
time: 6.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.80 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.5908589, upper bound: 143.5968149
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6311349, upper bound: 143.6348961
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6538692, upper bound: 143.6615222
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6545721, upper bound: 143.6623154
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6630762, upper bound: 143.6643727
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6869825, upper bound: 143.6919673
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.5810630, upper bound: 143.5860475
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6215539, upper bound: 143.6238746
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6303346, upper bound: 143.6330331
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6530526, upper bound: 143.6587880
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6428470, upper bound: 143.6490374
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6809016, upper bound: 143.6843873
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6179748, upper bound: 143.6262535
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6158864, upper bound: 143.6245780
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6388698, upper bound: 143.6464926
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6383727, upper bound: 143.6461264
IS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6031199, upper bound: 143.6101436
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.80
Output dim: 4, lower bound: -143.6378572, upper bound: 143.6493985
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -143.6625824, upper bound: 143.6726270
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -143.6304012, upper bound: 143.6304012
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6649315, upper bound: 143.6590435
time: 7.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6417354, upper bound: 143.6417354
time: 5.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.96
Output dim: 4, lower bound: -143.6649315, upper bound: 143.6590435
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.96
Output dim: 4, lower bound: -143.6417354, upper bound: 143.6417354

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -75.7754288, 60.4595566, -76.0228043, 60.6544914, -136.4299011, 136.4823456
1: -62.7865219, 53.6056404, -62.9943390, 53.7792549, -116.5657806, 116.5999756
2: -83.1312408, 54.2274513, -83.4031906, 54.4029045, -137.5341492, 137.6306458
3: -88.4644012, 46.7679405, -88.7539597, 46.9208794, -135.3852844, 135.5218964
4: -81.7224503, 62.8103371, -81.9910431, 63.0132256, -144.7356415, 144.8013763
5: -72.9203491, 56.7705574, -73.1580811, 56.9551659, -129.8755188, 129.9286194
6: -70.0489120, 66.8218460, -70.2768860, 67.0410385, -137.0899506, 137.0987244
7: -75.7611084, 63.7964745, -76.0090866, 64.0053253, -139.7664185, 139.8055573
8: -91.7253265, 63.5594482, -92.0233536, 63.7623940, -155.4877167, 155.5827942
9: -69.1330185, 67.9258499, -69.3594666, 68.1468353, -137.2798462, 137.2852783

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5611779, upper bound: 143.5582645
time: 7.25 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6042710, upper bound: 143.6012072
time: 7.91 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6591245, upper bound: 143.6533346
time: 7.45 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -97.5423889, 77.7365036, -75.3719025, 60.1434021, -157.6857758, 153.1083984
1: -81.1014328, 68.8907394, -62.4460297, 53.3217430, -134.4231720, 131.3367615
2: -106.9554520, 69.4951782, -82.6866074, 53.9403191, -160.8957672, 152.1817932
3: -113.8497009, 60.1707802, -87.9906235, 46.5180473, -160.3677368, 148.1614075
4: -105.0295410, 80.6163635, -81.2843552, 62.4789162, -167.5084534, 161.9007263
5: -93.9048538, 72.9319458, -72.5331955, 56.4702110, -150.3750610, 145.4651489
6: -89.9386978, 86.1439285, -69.6767349, 66.4631271, -156.4018097, 155.8206635
7: -97.3581924, 82.0536880, -75.3552322, 63.4553146, -160.8135071, 157.4089050
8: -117.8584747, 81.4248352, -91.2375107, 63.2285652, -181.0870361, 172.6623077
9: -88.8761597, 87.3950424, -68.7620773, 67.5658798, -156.4420166, 156.1571198

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6043070, upper bound: 143.6060369
time: 5.58 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6360029, upper bound: 143.6360029
time: 5.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.58 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 25.58
Output dim: 4, lower bound: -143.6042710, upper bound: 143.6012072
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 25.58
Output dim: 4, lower bound: -143.6591245, upper bound: 143.6533346
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 25.58
Output dim: 4, lower bound: -143.6043070, upper bound: 143.6060369
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 25.58
Output dim: 4, lower bound: -143.6360029, upper bound: 143.6360029

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -69.6014862, 55.5721283, -75.0562973, 59.8893547, -129.4908447, 130.6284180
1: -57.5442352, 49.2468033, -62.1738701, 53.0968056, -110.6410370, 111.4206696
2: -76.2999649, 49.8429146, -82.3339157, 53.7164612, -130.0164185, 132.1768341
3: -81.2436676, 42.9487839, -87.6237564, 46.3229675, -127.5666351, 130.5725403
4: -75.0962753, 57.7355690, -80.9539642, 62.2187538, -137.3150330, 138.6895294
5: -66.9880066, 52.1712303, -72.2294388, 56.2351303, -123.2231369, 124.4006653
6: -64.3609085, 61.3479958, -69.3863754, 66.1842422, -130.5451508, 130.7343750
7: -69.5795670, 58.5862923, -75.0414658, 63.1896706, -132.7692108, 133.6277466
8: -84.2888260, 58.4792328, -90.8593063, 62.9672394, -147.2560577, 149.3385315
9: -63.4707680, 62.3343391, -68.4729309, 67.2716141, -130.7423706, 130.8072662

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5532604, upper bound: 143.5505580
time: 7.72 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6218093, upper bound: 143.6167587
time: 7.31 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6215558, upper bound: 143.6150368
time: 8.22 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6536557, upper bound: 143.6478474
time: 6.03 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -92.7847443, 73.9378281, -74.4557419, 59.4117165, -152.1964417, 148.3935699
1: -77.0551910, 65.5191956, -61.6666145, 52.6722412, -129.7274323, 127.1858063
2: -101.6757355, 66.1023636, -81.6698456, 53.2871246, -154.9628601, 147.7722168
3: -108.2916336, 57.2319260, -86.9202042, 45.9520912, -154.2437134, 144.1521301
4: -99.8842316, 76.6764984, -80.2934647, 61.7204514, -161.6046753, 156.9699707
5: -89.3289337, 69.3778458, -71.6516953, 55.7859383, -145.1148682, 141.0295105
6: -85.5430298, 81.9075851, -68.8303680, 65.6469193, -151.1899414, 150.7379456
7: -92.5772018, 78.0383911, -74.4347992, 62.6819992, -155.2592010, 152.4731903
8: -112.0712662, 77.4210510, -90.1228485, 62.4574814, -174.5287170, 167.5438843
9: -84.4948578, 83.0564651, -67.9184189, 66.7302933, -151.2251587, 150.9748840

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6080573, upper bound: 143.6071179
time: 6.40 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6046223, upper bound: 143.6046223
time: 5.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.28 seconds
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 4, lower bound: -143.6215558, upper bound: 143.6150368
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.28
Output dim: 4, lower bound: -143.6536557, upper bound: 143.6478474
IS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 29.28
Output dim: 4, lower bound: -143.6080573, upper bound: 143.6071179
IS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 29.28
Output dim: 4, lower bound: -143.6046223, upper bound: 143.6046223

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -60.1921425, 48.0266457, -54.1093216, 43.0580673, -103.2502136, 102.1359711
1: -49.5463829, 42.5732384, -44.3531303, 38.2283821, -87.7747650, 86.9263687
2: -65.8350601, 43.1281242, -59.0175743, 38.7437057, -104.5787659, 102.1456985
3: -70.2417374, 37.1330986, -63.1709328, 33.3657455, -103.6074753, 100.3040314
4: -64.8974838, 49.9110641, -58.2614861, 44.7559204, -109.6534042, 108.1725464
5: -57.9308662, 45.1230888, -52.0762177, 40.5297394, -98.4606018, 97.1993027
6: -55.6722679, 52.9649506, -50.0372009, 47.5170021, -103.1892700, 103.0021515
7: -60.1176567, 50.6422882, -53.9902649, 45.4923630, -105.6100159, 104.6325531
8: -72.7970810, 50.4812851, -65.2306747, 45.0668983, -117.8639755, 115.7119598
9: -54.8002014, 53.7192802, -49.1485901, 48.0444603, -102.8446579, 102.8678589

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6088700
time: 7.55 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6150368
time: 7.81 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -68.6913910, 54.8447533, -70.2233658, 56.0304489, -124.7218323, 125.0681152
1: -56.7701378, 48.6021652, -58.0620766, 49.6714935, -106.4416351, 106.6642227
2: -75.2887421, 49.1937599, -76.9707794, 50.2709312, -125.5596771, 126.1645279
3: -80.1793442, 42.3870239, -81.9777756, 43.3372650, -123.5166092, 124.3647842
4: -74.1113510, 56.9816933, -75.7273102, 58.2185173, -132.3298645, 132.7089996
5: -66.1118317, 51.4911728, -67.5798874, 52.6260071, -118.7378387, 119.0710602
6: -63.5200577, 60.5366554, -64.9224091, 61.8794899, -125.3995514, 125.4590607
7: -68.6644058, 57.8179817, -70.1859283, 59.1108818, -127.7752838, 128.0039062
8: -83.1810226, 57.7130318, -84.9802246, 58.9008827, -142.0819092, 142.6932526
9: -62.6321259, 61.5033379, -64.0223846, 62.8635902, -125.4957123, 125.5257263

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6194041
time: 8.43 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6478474
time: 8.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.35 seconds
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 20.35
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6088700
IS_A1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 20.35
Output dim: 4, lower bound: -143.6142211, upper bound: 143.6150368
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 20.35
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6194041
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.35
Output dim: 4, lower bound: -143.6226807, upper bound: 143.6478474

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -48.9253693, 38.9495392, -70.2233658, 56.0304489, -104.9558182, 109.1728973
1: -39.9948921, 34.5751038, -58.0620766, 49.6714935, -89.6663818, 92.6371765
2: -53.2756615, 35.0823135, -76.9707794, 50.2709312, -103.5465927, 112.0530930
3: -57.0935478, 30.1711178, -81.9777756, 43.3372650, -100.4308167, 112.1488953
4: -52.7179947, 40.4957924, -75.7273102, 58.2185173, -110.9365082, 116.2230988
5: -47.1085320, 36.6577263, -67.5798874, 52.6260071, -99.7345428, 104.2376099
6: -45.2901878, 42.9477539, -64.9224091, 61.8794899, -107.1696777, 107.8701630
7: -48.7971802, 41.1083984, -70.1859283, 59.1108818, -107.9080505, 111.2943268
8: -58.9783325, 40.8071785, -84.9802246, 58.9008827, -117.8792114, 125.7873993
9: -44.4083176, 43.3711090, -64.0223846, 62.8635902, -107.2719116, 107.3934937

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5830405, upper bound: 143.5892827
time: 7.99 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6086069, upper bound: 143.6134763
time: 7.26 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -64.7955780, 51.7317848, -70.2233658, 56.0304489, -120.8260040, 121.9551544
1: -53.4576569, 45.8433304, -58.0620766, 49.6714935, -103.1291504, 103.9054108
2: -70.9617386, 46.4162064, -76.9707794, 50.2709312, -121.2326660, 123.3869858
3: -75.6238785, 39.9820251, -81.9777756, 43.3372650, -118.9611435, 121.9597931
4: -69.8961029, 53.7560272, -75.7273102, 58.2185173, -128.1146240, 129.4833374
5: -62.3621979, 48.5795937, -67.5798874, 52.6260071, -114.9882050, 116.1594849
6: -59.9222946, 57.0658646, -64.9224091, 61.8794899, -121.8017807, 121.9882660
7: -64.7473068, 54.5286636, -70.1859283, 59.1108818, -123.8581848, 124.7145920
8: -78.4421463, 54.4347458, -84.9802246, 58.9008827, -137.3430328, 139.4149780
9: -59.0426178, 57.9482040, -64.0223846, 62.8635902, -121.9062042, 121.9705887

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5897358, upper bound: 143.6260278
time: 8.53 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5825059, upper bound: 143.6181681
time: 9.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 40.54 seconds
IS_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 40.54
Output dim: 4, lower bound: -143.5830405, upper bound: 143.5892827
IS_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 40.54
Output dim: 4, lower bound: -143.6086069, upper bound: 143.6134763
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 40.54
Output dim: 4, lower bound: -143.5897358, upper bound: 143.6260278
IS_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 40.54
Output dim: 4, lower bound: -143.5825059, upper bound: 143.6181681

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -62.2472801, 49.7036285, -57.6382790, 46.0151749, -108.2624512, 107.3418961
1: -51.3141098, 44.0522003, -47.4716949, 40.8250618, -92.1391754, 91.5238953
2: -68.1471100, 44.6006966, -63.0678406, 41.3031883, -109.4503021, 107.6685181
3: -72.6302414, 38.4240532, -67.1931992, 35.6435051, -108.2737350, 105.6172409
4: -67.1568222, 51.6689262, -62.1924133, 47.9130211, -115.0698242, 113.8613434
5: -59.9010963, 46.6725502, -55.4231682, 43.2120056, -103.1130753, 102.0957184
6: -57.5709076, 54.8162079, -53.3056717, 50.7634811, -108.3343887, 108.1218796
7: -62.1968994, 52.3752441, -57.5917969, 48.4770432, -110.6739426, 109.9670410
8: -75.3762741, 52.3408470, -69.8353806, 48.5643578, -123.9406281, 122.1762238
9: -56.7151070, 55.6637077, -52.5275879, 51.5761986, -108.2913055, 108.1912994

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681
time: 7.90 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681
time: 8.30 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.77 seconds
IS_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 19.77
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681
IS_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 19.77
Output dim: 4, lower bound: -143.6246378, upper bound: 143.6181681

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -52.4100266, 41.8666000, -57.6382790, 46.0151749, -98.4252014, 99.5048752
1: -43.0761375, 37.1460266, -47.4716949, 40.8250618, -83.9011993, 84.6177216
2: -57.2795563, 37.5978012, -63.0678406, 41.3031883, -98.5827484, 100.6656418
3: -61.0660591, 32.4149246, -67.1931992, 35.6435051, -96.7095490, 99.6081085
4: -56.5757446, 43.6117973, -62.1924133, 47.9130211, -104.4887619, 105.8042068
5: -50.4002914, 39.3099976, -55.4231682, 43.2120056, -93.6122818, 94.7331696
6: -48.4932442, 46.1533203, -53.3056717, 50.7634811, -99.2567291, 99.4589920
7: -52.3478088, 44.0692749, -57.5917969, 48.4770432, -100.8248520, 101.6610718
8: -63.5343361, 44.2504311, -69.8353806, 48.5643578, -112.0986938, 114.0858154
9: -47.7454872, 46.8593788, -52.5275879, 51.5761986, -99.3216858, 99.3869629

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5865119, upper bound: 143.5819898
time: 8.48 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
time: 9.15 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308
time: 7.30 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -49.3865967, 39.4502754, -57.6382790, 46.0151749, -95.4017715, 97.0885468
1: -40.5630302, 35.0370331, -47.4716949, 40.8250618, -81.3880844, 82.5087280
2: -53.9154205, 35.4234886, -63.0678406, 41.3031883, -95.2186127, 98.4913254
3: -57.5452614, 30.6224766, -67.1931992, 35.6435051, -93.1887665, 97.8156738
4: -53.3789139, 41.1305809, -62.1924133, 47.9130211, -101.2919235, 103.3229904
5: -47.5030251, 37.0650330, -55.4231682, 43.2120056, -90.7150192, 92.4881973
6: -45.7314301, 43.5351028, -53.3056717, 50.7634811, -96.4949112, 96.8407669
7: -49.3551788, 41.5209198, -57.5917969, 48.4770432, -97.8322220, 99.1127167
8: -59.9189529, 41.7493744, -69.8353806, 48.5643578, -108.4833069, 111.5847549
9: -45.0488167, 44.1828194, -52.5275879, 51.5761986, -96.6250153, 96.7104034

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5865119, upper bound: 143.5819898
time: 8.55 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
time: 9.19 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308
time: 6.84 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 47.92 seconds
IS_A1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 47.92
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
IS_A1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 47.92
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308
IS_A1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 47.92
Output dim: 4, lower bound: -143.5900305, upper bound: 143.5844942
IS_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 47.92
Output dim: 4, lower bound: -143.6263022, upper bound: 143.6207308

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -51.5796356, 41.2058372, -53.7174149, 42.8914986, -94.4711304, 94.9232330
1: -42.3723373, 36.5594673, -44.1440277, 38.0519714, -80.4243088, 80.7034912
2: -56.3609505, 37.0142937, -58.7353630, 38.5423241, -94.9032745, 95.7496338
3: -60.1019325, 31.9029083, -62.6337204, 33.2080994, -93.3100281, 94.5366287
4: -55.6826210, 42.9229774, -57.9612160, 44.6611633, -100.3437729, 100.8841858
5: -49.6057892, 38.6919327, -51.6681786, 40.2961655, -89.9019547, 90.3600845
6: -47.7285881, 45.4170570, -49.6778374, 47.2833099, -95.0118866, 95.0948944
7: -51.5163727, 43.3731918, -53.6615829, 45.1887932, -96.7051620, 97.0347519
8: -62.5217285, 43.5426941, -65.0578842, 45.2283325, -107.7500610, 108.6005783
9: -46.9862099, 46.1074448, -48.9375000, 48.0245438, -95.0107574, 95.0449448

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6112682
time: 7.62 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6318373
time: 7.48 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -48.5937805, 38.8181725, -53.7174149, 42.8914986, -91.4852753, 92.5355835
1: -39.8905258, 34.4759903, -44.1440277, 38.0519714, -77.9424973, 78.6200180
2: -53.0376625, 34.8658371, -58.7353630, 38.5423241, -91.5799866, 93.6011810
3: -56.6258011, 30.1339054, -62.6337204, 33.2080994, -89.8339005, 92.7676163
4: -52.5275803, 40.4718018, -57.9612160, 44.6611633, -97.1887360, 98.4330063
5: -46.7441483, 36.4738045, -51.6681786, 40.2961655, -87.0403061, 88.1419754
6: -45.0008698, 42.8316689, -49.6778374, 47.2833099, -92.2841797, 92.5095062
7: -48.5622406, 40.8546677, -53.6615829, 45.1887932, -93.7510376, 94.5162430
8: -58.9503937, 41.0730629, -65.0578842, 45.2283325, -104.1787262, 106.1309357
9: -44.3229332, 43.4647255, -48.9375000, 48.0245438, -92.3474731, 92.4022217

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5859774, upper bound: 143.5830211
time: 8.08 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5859774, upper bound: 143.6207308
time: 8.50 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 30.31 seconds
IS_A1_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 30.31
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6112682
IS_A1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 30.31
Output dim: 4, lower bound: -143.6154372, upper bound: 143.6318373
IS_A1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 30.31
Output dim: 4, lower bound: -143.5859774, upper bound: 143.5830211
IS_A1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 30.31
Output dim: 4, lower bound: -143.5859774, upper bound: 143.6207308

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -48.6060562, 38.8379021, -53.7174149, 42.8914986, -91.4975510, 92.5553055
1: -39.8501663, 34.4563637, -44.1440277, 38.0519714, -77.9021378, 78.6003799
2: -53.0693130, 34.9242706, -58.7353630, 38.5423241, -91.6116333, 93.6596298
3: -56.6526566, 30.0714684, -62.6337204, 33.2080994, -89.8607559, 92.7051773
4: -52.4883156, 40.4533119, -57.9612160, 44.6611633, -97.1494675, 98.4145203
5: -46.7606239, 36.4757004, -51.6681786, 40.2961655, -87.0567780, 88.1438675
6: -44.9910164, 42.7791290, -49.6778374, 47.2833099, -92.2743225, 92.4569702
7: -48.5422668, 40.8767815, -53.6615829, 45.1887932, -93.7310638, 94.5383606
8: -58.8920021, 41.0069656, -65.0578842, 45.2283325, -104.1203308, 106.0648499
9: -44.2650642, 43.4144592, -48.9375000, 48.0245438, -92.2896042, 92.3519592

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6094533, upper bound: 143.6299096
time: 7.57 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6073386, upper bound: 143.6291295
time: 8.27 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -45.7414055, 36.5441017, -53.7174149, 42.8914986, -88.6329041, 90.2615128
1: -37.4723473, 32.4578171, -44.1440277, 38.0519714, -75.5243225, 76.6018295
2: -49.8767242, 32.8612862, -58.7353630, 38.5423241, -88.4190521, 91.5966339
3: -53.3171234, 28.3745461, -62.6337204, 33.2080994, -86.5252228, 91.0082474
4: -49.4631157, 38.0994072, -57.9612160, 44.6611633, -94.1242828, 96.0606155
5: -44.0129509, 34.3447800, -51.6681786, 40.2961655, -84.3091049, 86.0129471
6: -42.3765144, 40.2961197, -49.6778374, 47.2833099, -89.6598053, 89.9739532
7: -45.7069893, 38.4557228, -53.6615829, 45.1887932, -90.8957825, 92.1173096
8: -55.4664497, 38.6428375, -65.0578842, 45.2283325, -100.6947708, 103.7007141
9: -41.7156143, 40.8792725, -48.9375000, 48.0245438, -89.7401581, 89.8167725

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5792537, upper bound: 143.6192116
time: 7.20 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.5784785, upper bound: 143.6190814
time: 7.77 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 39.77 seconds
IS_A1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 39.77
Output dim: 4, lower bound: -143.6094533, upper bound: 143.6299096
IS_A1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 39.77
Output dim: 4, lower bound: -143.6073386, upper bound: 143.6291295
IS_A1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 39.77
Output dim: 4, lower bound: -143.5792537, upper bound: 143.6192116
IS_A1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 39.77
Output dim: 4, lower bound: -143.5784785, upper bound: 143.6190814

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -46.1245193, 36.8651047, -47.2441216, 37.7457848, -83.8702927, 84.1092148
1: -37.7774048, 32.7082253, -38.7313232, 33.4896126, -71.2670135, 71.4395447
2: -50.3197594, 33.1623993, -51.5665398, 33.9424400, -84.2621994, 84.7289429
3: -53.7177238, 28.5606709, -54.9725685, 29.2602215, -82.9779434, 83.5332260
4: -49.8040504, 38.4064827, -50.9525070, 39.3241577, -89.1281891, 89.3589706
5: -44.3674049, 34.6048622, -45.4236870, 35.4227524, -79.7901611, 80.0285339
6: -42.7047882, 40.5870209, -43.7025108, 41.5654259, -84.2702179, 84.2895355
7: -46.0361786, 38.7668686, -47.1220551, 39.6931534, -85.7293320, 85.8889160
8: -55.9005623, 38.9793892, -57.2579765, 39.9359207, -95.8364868, 96.2373581
9: -41.9949646, 41.1811829, -43.0165558, 42.2014503, -84.1964111, 84.1977386

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6289552
time: 7.40 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6291295
time: 7.99 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -46.8188171, 37.4156837, -51.6063576, 41.1996994, -88.0185165, 89.0220261
1: -38.3574066, 33.1970329, -42.3659592, 36.5583687, -74.9157715, 75.5629883
2: -51.0885391, 33.6527176, -56.3907814, 37.0173569, -88.1058960, 90.0434952
3: -54.5414047, 28.9828091, -60.1477509, 31.9251728, -86.4665756, 89.1305618
4: -50.5574760, 38.9793320, -55.6821709, 42.9123993, -93.4698792, 94.6614990
5: -45.0378036, 35.1253281, -49.6265526, 38.6712799, -83.7090836, 84.7518768
6: -43.3463631, 41.2006493, -47.7560081, 45.4170227, -88.7633820, 88.9566422
7: -46.7391472, 39.3568077, -51.5199890, 43.3656387, -90.1047821, 90.8767929
8: -56.7392311, 39.5461731, -62.5396614, 43.5086784, -100.2479019, 102.0858307
9: -42.6311188, 41.8057861, -46.9965286, 46.1142769, -88.7453918, 88.8023148

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6289552
time: 8.05 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6351532, upper bound: 143.6291295
time: 7.26 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.2399445, 34.5522118, -47.2441216, 37.7457848, -80.9857330, 81.7963333
1: -35.3819504, 30.6967182, -38.7313232, 33.4896126, -68.8715591, 69.4280396
2: -47.0997391, 31.0885563, -51.5665398, 33.9424400, -81.0421753, 82.6550980
3: -50.3588676, 26.8540096, -54.9725685, 29.2602215, -79.6190872, 81.8265610
4: -46.7536278, 36.0379524, -50.9525070, 39.3241577, -86.0777817, 86.9904556
5: -41.5977974, 32.4550209, -45.4236870, 35.4227524, -77.0205536, 77.8787079
6: -40.0709572, 38.0890045, -43.7025108, 41.5654259, -81.6363831, 81.7915192
7: -43.1736069, 36.3281631, -47.1220551, 39.6931534, -82.8667450, 83.4502182
8: -52.4460106, 36.5999641, -57.2579765, 39.9359207, -92.3819275, 93.8579407
9: -39.4283257, 38.6300430, -43.0165558, 42.2014503, -81.6297760, 81.6465988

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7353923, upper bound: 143.7350238
time: 8.18 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
time: 6.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.28
Output dim: 4, lower bound: -143.7353923, upper bound: 143.7350238
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.28
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -66.9062042, 53.4514427, -70.4277115, 56.2345390, -123.1407394, 123.8791428
1: -55.2424507, 47.3330040, -58.2350349, 49.8213501, -105.0637970, 105.5680389
2: -73.2837601, 47.8946266, -77.1947021, 50.4089394, -123.6927032, 125.0893250
3: -78.1033096, 41.2522545, -82.2189789, 43.4409561, -121.5442657, 123.4712067
4: -72.2304153, 55.5285950, -76.0003510, 58.4195251, -130.6499329, 131.5289459
5: -64.4123077, 50.2072067, -67.7918854, 52.8146172, -117.2269211, 117.9990845
6: -61.8486290, 58.9683495, -65.1053162, 62.0867844, -123.9354095, 124.0736542
7: -66.8461456, 56.3229561, -70.3863602, 59.2916985, -126.1378250, 126.7093124
8: -81.0043411, 56.2406425, -85.2609863, 59.1454086, -140.1497498, 141.5016327
9: -61.0297775, 59.9055862, -64.2485809, 63.0913506, -124.1211166, 124.1541519

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 155

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
time: 7.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
time: 6.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -58.7513885, 46.9397469, -58.0478630, 46.4383392, -105.1897202, 104.9875793
1: -48.2426414, 41.4969864, -47.7464752, 41.0816422, -89.3242798, 89.2434540
2: -64.1473160, 41.9993095, -63.4714317, 41.5932579, -105.7405701, 105.4707336
3: -68.5663376, 36.1149368, -67.7501755, 35.7675056, -104.3338318, 103.8651123
4: -63.5618553, 48.8138351, -62.7405739, 48.2484436, -111.8103027, 111.5544052
5: -56.5759201, 44.1477852, -55.9089241, 43.6322098, -100.2081146, 100.0567093
6: -54.2838554, 51.7676506, -53.6839600, 51.1420135, -105.4258728, 105.4516144
7: -58.6106453, 49.3955307, -57.9578857, 48.8589096, -107.4695435, 107.3533936
8: -71.1166306, 49.4256859, -70.3106308, 48.9079170, -120.0245514, 119.7363129
9: -53.6192398, 52.4673309, -52.9470367, 51.8988228, -105.5180664, 105.4143677

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
time: 8.11 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
time: 7.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.95 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.95
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.95
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.95
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.95
Output dim: 4, lower bound: -143.7338807, upper bound: 143.7338807

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -66.9062042, 53.4514427, -66.9062042, 53.4514427, -120.3576355, 120.3576355
1: -55.2424507, 47.3330040, -55.2424507, 47.3330040, -102.5754547, 102.5754547
2: -73.2837601, 47.8946266, -73.2837601, 47.8946266, -121.1783905, 121.1783905
3: -78.1033096, 41.2522545, -78.1033096, 41.2522545, -119.3555374, 119.3555374
4: -72.2304153, 55.5285950, -72.2304153, 55.5285950, -127.7590103, 127.7590103
5: -64.4123077, 50.2072067, -64.4123077, 50.2072067, -114.6195145, 114.6195145
6: -61.8486290, 58.9683495, -61.8486290, 58.9683495, -120.8169632, 120.8169632
7: -66.8461456, 56.3229561, -66.8461456, 56.3229561, -123.1690979, 123.1690979
8: -81.0043411, 56.2406425, -81.0043411, 56.2406425, -137.2449799, 137.2449799
9: -61.0297775, 59.9055862, -61.0297775, 59.9055862, -120.9353409, 120.9353409

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7177787, upper bound: 143.7171605
time: 7.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7174384, upper bound: 143.7169364
time: 9.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -66.9062042, 53.4514427, -58.7513885, 46.9397469, -113.8459320, 112.2028046
1: -55.2424507, 47.3330040, -48.2426414, 41.4969864, -96.7394409, 95.5756378
2: -73.2837601, 47.8946266, -64.1473160, 41.9993095, -115.2830658, 112.0419464
3: -78.1033096, 41.2522545, -68.5663376, 36.1149368, -114.2182465, 109.8185654
4: -72.2304153, 55.5285950, -63.5618553, 48.8138351, -121.0442429, 119.0904465
5: -64.4123077, 50.2072067, -56.5759201, 44.1477852, -108.5600891, 106.7831268
6: -61.8486290, 58.9683495, -54.2838554, 51.7676506, -113.6162796, 113.2522049
7: -66.8461456, 56.3229561, -58.6106453, 49.3955307, -116.2416687, 114.9336014
8: -81.0043411, 56.2406425, -71.1166306, 49.4256859, -130.4300232, 127.3572693
9: -61.0297775, 59.9055862, -53.6192398, 52.4673309, -113.4970932, 113.5248260

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 155

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7177787, upper bound: 143.7171605
time: 10.20 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7174384, upper bound: 143.7169364
time: 9.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -58.7513885, 46.9397469, -66.7878113, 53.3632278, -112.1146011, 113.7275543
1: -48.2426414, 41.4969864, -55.1080322, 47.2443275, -95.4869537, 96.6050186
2: -64.1473160, 41.9993095, -73.1583252, 47.7980576, -111.9453735, 115.1576309
3: -68.5663376, 36.1149368, -77.9681778, 41.1862068, -109.7525406, 114.0831146
4: -63.5618553, 48.8138351, -72.1211700, 55.4252625, -118.9871216, 120.9349976
5: -56.5759201, 44.1477852, -64.2936478, 50.1124611, -106.6883698, 108.4414368
6: -54.2838554, 51.7676506, -61.7469902, 58.8502426, -113.1340942, 113.5146408
7: -58.6106453, 49.3955307, -66.7307663, 56.2090988, -114.8197479, 116.1262894
8: -71.1166306, 49.4256859, -80.8590622, 56.1526947, -127.2693253, 130.2847443
9: -53.6192398, 52.4673309, -60.9104309, 59.7621918, -113.3814316, 113.3777618

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7160700, upper bound: 143.7163450
time: 8.19 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7154991, upper bound: 143.7154991
time: 7.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -58.7513885, 46.9397469, -58.7513885, 46.9397469, -105.6911087, 105.6911087
1: -48.2426414, 41.4969864, -48.2426414, 41.4969864, -89.7396164, 89.7396240
2: -64.1473160, 41.9993095, -64.1473160, 41.9993095, -106.1466217, 106.1466217
3: -68.5663376, 36.1149368, -68.5663376, 36.1149368, -104.6812744, 104.6812744
4: -63.5618553, 48.8138351, -63.5618553, 48.8138351, -112.3756866, 112.3756866
5: -56.5759201, 44.1477852, -56.5759201, 44.1477852, -100.7237091, 100.7237091
6: -54.2838554, 51.7676506, -54.2838554, 51.7676506, -106.0515060, 106.0515060
7: -58.6106453, 49.3955307, -58.6106453, 49.3955307, -108.0061722, 108.0061722
8: -71.1166306, 49.4256859, -71.1166306, 49.4256859, -120.5423126, 120.5423126
9: -53.6192398, 52.4673309, -53.6192398, 52.4673309, -106.0865707, 106.0865707

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7163450, upper bound: 143.7160700
time: 9.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7154991, upper bound: 143.7154991
time: 8.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.58 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7177787, upper bound: 143.7171605
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7174384, upper bound: 143.7169364
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7177787, upper bound: 143.7171605
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7174384, upper bound: 143.7169364
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7160700, upper bound: 143.7163450
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7154991, upper bound: 143.7154991
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7163450, upper bound: 143.7160700
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.58
Output dim: 4, lower bound: -143.7154991, upper bound: 143.7154991

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -54.4732056, 43.5502396, -58.2657089, 46.5734901, -101.0466766, 101.8159409
1: -44.8182678, 38.5980415, -47.9847069, 41.2605743, -86.0788345, 86.5827484
2: -59.5511398, 39.0395851, -63.7420998, 41.7375298, -101.2886658, 102.7816849
3: -63.4910812, 33.6505356, -67.9483337, 35.9691696, -99.4602509, 101.5988541
4: -58.8591805, 45.3506927, -62.9437218, 48.4501648, -107.3093414, 108.2944183
5: -52.3978767, 40.9078369, -56.0623055, 43.7421494, -96.1400299, 96.9701385
6: -50.3696213, 48.0092010, -53.8722649, 51.3467865, -101.7164078, 101.8814621
7: -54.4027634, 45.8191414, -58.1996078, 49.0193748, -103.4221344, 104.0187531
8: -66.0453949, 46.0229721, -70.6075974, 49.1417160, -115.1871109, 116.6305695
9: -49.6862221, 48.7756882, -53.1428795, 52.1589737, -101.8451996, 101.9185638

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7333567, upper bound: 143.7331933
time: 10.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7341798, upper bound: 143.7340617
time: 9.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -51.9445267, 41.5257149, -52.6096153, 42.0676880, -94.0122147, 94.1353302
1: -42.7164421, 36.8347321, -43.2699585, 37.2945824, -80.0110245, 80.1046906
2: -56.7357712, 37.2138748, -57.4879951, 37.7243347, -94.4600906, 94.7018738
3: -60.5441208, 32.1468086, -61.3192673, 32.4944038, -93.0385284, 93.4660645
4: -56.1850471, 43.2780952, -56.8709259, 43.8257751, -100.0108109, 100.1490173
5: -49.9742699, 39.0375824, -50.6148415, 39.5270538, -89.5013123, 89.6524200
6: -48.0486870, 45.8202515, -48.6720581, 46.3735237, -94.4222107, 94.4923019
7: -51.9001503, 43.6928864, -52.5408821, 44.2648087, -96.1649628, 96.2337646
8: -63.0345726, 43.9365044, -63.8104248, 44.4809532, -107.5155258, 107.7469177
9: -47.4403305, 46.5396042, -48.0014153, 47.1043091, -94.5446396, 94.5410156

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7330016, upper bound: 143.7329669
time: 8.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7338512, upper bound: 143.7338512
time: 7.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -54.4732056, 43.5502396, -50.6272850, 40.4624557, -94.9356461, 94.1775208
1: -44.8182678, 38.5980415, -41.4671783, 35.7891464, -80.6073990, 80.0652161
2: -59.5511398, 39.0395851, -55.1604080, 36.2101440, -95.7612762, 94.1999969
3: -63.4910812, 33.6505356, -59.0329323, 31.1638527, -94.6549301, 92.6834564
4: -58.8591805, 45.3506927, -54.8339272, 42.1687737, -101.0279541, 100.1846161
5: -52.3978767, 40.9078369, -48.7339783, 38.0679398, -90.4658203, 89.6418152
6: -50.3696213, 48.0092010, -46.7903404, 44.6261978, -94.9958191, 94.7995453
7: -54.4027634, 45.8191414, -50.4840698, 42.5388985, -96.9416428, 96.3031998
8: -66.0453949, 46.0229721, -61.3322678, 42.7367058, -108.7821045, 107.3552399
9: -49.6862221, 48.7756882, -46.2205963, 45.2293968, -94.9156189, 94.9962769

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7035001, upper bound: 143.7027145
time: 8.88 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7037381, upper bound: 143.7029257
time: 10.76 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -51.9445267, 41.5257149, -44.5073509, 35.5864296, -87.5309448, 86.0330582
1: -42.7164421, 36.8347321, -36.3648262, 31.4955807, -74.2120209, 73.1995544
2: -56.7357712, 37.2138748, -48.3491135, 31.8660908, -88.6018524, 85.5629883
3: -60.5441208, 32.1468086, -51.8737946, 27.4457874, -87.9899063, 84.0205994
4: -56.1850471, 43.2780952, -48.2749214, 37.1524963, -93.3375168, 91.5530167
5: -49.9742699, 39.0375824, -42.8347893, 33.4887505, -83.4630203, 81.8723755
6: -48.0486870, 45.8202515, -41.1850853, 39.2390862, -87.2877655, 87.0053329
7: -51.9001503, 43.6928864, -44.3550262, 37.3633003, -89.2634506, 88.0479126
8: -63.0345726, 43.9365044, -53.9412079, 37.6987839, -100.7333527, 97.8777161
9: -47.4403305, 46.5396042, -40.6661568, 39.7554359, -87.1957550, 87.2057648

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7031499, upper bound: 143.7024795
time: 9.74 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7034695, upper bound: 143.7027621
time: 10.55 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -50.6272850, 40.4624557, -54.4721184, 43.5493469, -94.1766357, 94.9345703
1: -41.4671783, 35.7891464, -44.8173866, 38.5972672, -80.0644455, 80.6065292
2: -55.1604080, 36.2101440, -59.5499077, 39.0388260, -94.1992340, 95.7600479
3: -59.0329323, 31.1638527, -63.4896965, 33.6498146, -92.6827393, 94.6535492
4: -54.8339272, 42.1687737, -58.8579903, 45.3497963, -100.1837234, 101.0267639
5: -48.7339783, 38.0679398, -52.3968048, 40.9070282, -89.6410065, 90.4647446
6: -46.7903404, 44.6261978, -50.3685875, 48.0082397, -94.7985611, 94.9947815
7: -50.4840698, 42.5388985, -54.4015884, 45.8182182, -96.3022842, 96.9404678
8: -61.3322678, 42.7367058, -66.0440598, 46.0220833, -107.3543549, 108.7807617
9: -46.2205963, 45.2293968, -49.6851730, 48.7746658, -94.9952621, 94.9145660

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7027145, upper bound: 143.7035001
time: 10.25 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7029257, upper bound: 143.7037381
time: 8.98 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -44.5073509, 35.5864296, -51.8192596, 41.4234161, -85.9307556, 87.4056778
1: -36.3648262, 31.4955807, -42.6158829, 36.7453842, -73.1102142, 74.1114655
2: -48.3491135, 31.8660908, -56.5946198, 37.1264992, -85.4756165, 88.4607086
3: -51.8737946, 27.4457874, -60.3852997, 32.0649910, -83.9387817, 87.8310776
4: -48.2749214, 37.1524963, -56.0478668, 43.1749306, -91.4498444, 93.2003555
5: -42.8347893, 33.4887505, -49.8521881, 38.9435425, -81.7783356, 83.3409424
6: -41.1850853, 39.2390862, -47.9303169, 45.7089615, -86.8940430, 87.1693802
7: -44.3550262, 37.3633003, -51.7648354, 43.5879402, -87.9429626, 89.1281357
8: -53.9412079, 37.6987839, -62.8802605, 43.8339119, -97.7751160, 100.5790329
9: -40.6661568, 39.7554359, -47.3194427, 46.4219742, -87.0881348, 87.0748749

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7024795, upper bound: 143.7031499
time: 10.06 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7027621, upper bound: 143.7034695
time: 8.96 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -47.0895844, 37.6399956, -50.6272850, 40.4624557, -87.5520325, 88.2672806
1: -38.5143127, 33.3032913, -41.4671783, 35.7891464, -74.3034439, 74.7704697
2: -51.2326584, 33.6920624, -55.1604080, 36.2101440, -87.4427948, 88.8524704
3: -54.8868217, 29.0205364, -59.0329323, 31.1638527, -86.0506744, 88.0534592
4: -51.0333862, 39.2751770, -54.8339272, 42.1687737, -93.2021561, 94.1091003
5: -45.3195419, 35.4123497, -48.7339783, 38.0679398, -83.3874817, 84.1463318
6: -43.5418320, 41.5130310, -46.7903404, 44.6261978, -88.1680298, 88.3033752
7: -46.9437256, 39.5393181, -50.4840698, 42.5388985, -89.4826202, 90.0233917
8: -57.0606461, 39.8277969, -61.3322678, 42.7367058, -99.7973480, 101.1600647
9: -43.0032578, 42.0728073, -46.2205963, 45.2293968, -88.2326508, 88.2933960

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 49

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7019539, upper bound: 143.7017073
time: 8.26 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7020744, upper bound: 143.7017830
time: 9.23 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.3056602, 34.6172485, -44.5073509, 35.5864296, -78.8920746, 79.1245880
1: -35.3713264, 30.6635036, -36.3648262, 31.4955807, -66.8669052, 67.0283279
2: -46.9926949, 30.9884415, -48.3491135, 31.8660908, -78.8587875, 79.3375549
3: -50.4824295, 26.7862720, -51.8737946, 27.4457874, -77.9282150, 78.6600647
4: -47.0026627, 36.1620598, -48.2749214, 37.1524963, -84.1551437, 84.4369736
5: -41.6716576, 32.5961113, -42.8347893, 33.4887505, -75.1604080, 75.4309006
6: -40.0815353, 38.2080002, -41.1850853, 39.2390862, -79.3205948, 79.3930740
7: -43.1623764, 36.3246880, -44.3550262, 37.3633003, -80.5256805, 80.6797104
8: -52.5086441, 36.7071381, -53.9412079, 37.6987839, -90.2074127, 90.6483459
9: -39.5979195, 38.7062340, -40.6661568, 39.7554359, -79.3533401, 79.3723907

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7018488, upper bound: 143.7020101
time: 7.23 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7016060, upper bound: 143.7016060
time: 6.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.75 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7333567, upper bound: 143.7331933
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7341798, upper bound: 143.7340617
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7330016, upper bound: 143.7329669
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7338512, upper bound: 143.7338512
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7035001, upper bound: 143.7027145
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7037381, upper bound: 143.7029257
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7031499, upper bound: 143.7024795
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7034695, upper bound: 143.7027621
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7027145, upper bound: 143.7035001
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7029257, upper bound: 143.7037381
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7024795, upper bound: 143.7031499
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7027621, upper bound: 143.7034695
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7019539, upper bound: 143.7017073
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7020744, upper bound: 143.7017830
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7018488, upper bound: 143.7020101
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.75
Output dim: 4, lower bound: -143.7016060, upper bound: 143.7016060

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -45.0830879, 36.0555420, -44.3473511, 35.4677277, -80.5508118, 80.4028778
1: -36.9259872, 31.9789295, -36.2893524, 31.4530029, -68.3789749, 68.2682800
2: -49.1389236, 32.3686295, -48.3152390, 31.8528862, -80.9918060, 80.6838531
3: -52.5391045, 27.8916874, -51.7118492, 27.4217510, -79.9608459, 79.6035385
4: -48.8180008, 37.6315460, -48.0493698, 37.0124855, -85.8304672, 85.6809158
5: -43.3911781, 33.9057236, -42.7099495, 33.3711853, -76.7623596, 76.6156616
6: -41.8092422, 39.7155838, -41.1782875, 39.0501823, -80.8594208, 80.8938675
7: -45.0323792, 37.9299850, -44.3061447, 37.3277893, -82.3601685, 82.2361298
8: -54.6560020, 38.1620331, -53.7400436, 37.4977798, -92.1537781, 91.9020691
9: -41.1147194, 40.3015938, -40.4400826, 39.5983620, -80.7130814, 80.7416687

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6903316, upper bound: 143.6910758
time: 9.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7281920, upper bound: 143.7280055
time: 8.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -43.9774323, 35.1866684, -47.1889267, 37.7240410, -81.7014618, 82.3755951
1: -35.9913979, 31.2020168, -38.6307297, 33.4432869, -69.4346848, 69.8327408
2: -47.9112930, 31.5931396, -51.4585152, 33.8421898, -81.7534790, 83.0516510
3: -51.2359924, 27.2230568, -55.0281906, 29.1651421, -80.4011154, 82.2512283
4: -47.6211510, 36.7215233, -51.1320648, 39.3503761, -86.9715271, 87.8535919
5: -42.3218613, 33.0673523, -45.4366264, 35.4759865, -77.7978516, 78.5039749
6: -40.7835236, 38.7401428, -43.7773933, 41.5608978, -82.3444061, 82.5175323
7: -43.9021416, 36.9789162, -47.1302032, 39.6818085, -83.5839386, 84.1091156
8: -53.3268051, 37.2862968, -57.1807785, 39.8810959, -93.2078857, 94.4670715
9: -40.0914192, 39.2946434, -43.0074005, 42.1104622, -82.2018814, 82.3020477

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6911738, upper bound: 143.6901291
time: 9.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7292412, upper bound: 143.7290917
time: 9.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -42.5941734, 34.0632210, -38.8718147, 31.1081619, -73.7023163, 72.9350128
1: -34.8627357, 30.2440834, -31.7306252, 27.6167545, -62.4794884, 61.9747086
2: -46.3558426, 30.5808640, -42.2402534, 27.9867706, -74.3426132, 72.8211212
3: -49.6504669, 26.4304161, -45.2985878, 24.0808964, -73.7313614, 71.7289886
4: -46.1878357, 35.5921059, -42.1821518, 32.5366898, -78.7245255, 77.7742538
5: -41.0052071, 32.0549278, -37.4333191, 29.2694321, -70.2746201, 69.4882507
6: -39.5405540, 37.5656967, -36.1655350, 34.2403069, -73.7808609, 73.7312241
7: -42.5698433, 35.8242226, -38.8248291, 32.7095299, -75.2793732, 74.6490479
8: -51.6894531, 36.1104889, -47.1580887, 32.9896011, -84.6790543, 83.2685623
9: -38.9087296, 38.1085510, -35.4730835, 34.7114677, -73.6201935, 73.5816269

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 64

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6898251, upper bound: 143.6907170
time: 9.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7278172, upper bound: 143.7277790
time: 8.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -41.9658661, 33.5791702, -42.1036491, 33.6735306, -75.6393967, 75.6828156
1: -34.3336258, 29.8067894, -34.4005241, 29.8801327, -64.2137451, 64.2073135
2: -45.6618156, 30.1466198, -45.8151474, 30.2509766, -75.9127960, 75.9617691
3: -48.9090919, 26.0547352, -49.0694542, 26.0577431, -74.9668350, 75.1241837
4: -45.5062256, 35.0793610, -45.6844940, 35.1889153, -80.6951370, 80.7638474
5: -40.3999672, 31.5793362, -40.5426788, 31.6666508, -72.0666122, 72.1220093
6: -38.9555016, 37.0147438, -39.1235580, 37.0959282, -76.0514297, 76.1382904
7: -41.9174538, 35.2805138, -42.0443344, 35.3976021, -77.3150406, 77.3248444
8: -50.9436684, 35.6274757, -51.0653954, 35.6813316, -86.6249847, 86.6928711
9: -38.3213158, 37.5321350, -38.3931122, 37.5708199, -75.8921356, 75.9252396

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 155

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6899517, upper bound: 143.6909324
time: 7.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7288723, upper bound: 143.7288723
time: 8.07 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -40.7997818, 32.6367340, -41.6807442, 33.3265266, -74.1263123, 74.3174744
1: -33.3278656, 28.9609814, -33.9534225, 29.4868755, -62.8147430, 62.9144058
2: -44.3777542, 29.3393917, -45.2049942, 29.8753090, -74.2530518, 74.5443878
3: -47.5355186, 25.2653408, -48.6069984, 25.6932392, -73.2287598, 73.8723373
4: -44.2310753, 34.1094704, -45.2669525, 34.8063469, -79.0374222, 79.3764191
5: -39.2795029, 30.7001381, -40.1516266, 31.3860283, -70.6655045, 70.8517532
6: -37.9101372, 35.9298935, -38.6695366, 36.7099571, -74.6200943, 74.5994263
7: -40.7492332, 34.3225822, -41.5498047, 35.0012436, -75.7504730, 75.8723907
8: -49.4661140, 34.5763130, -50.4650269, 35.2543869, -84.7204971, 85.0413208
9: -37.2086525, 36.4367447, -38.0800781, 37.1440315, -74.3526764, 74.5168228

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6880362, upper bound: 143.6875134
time: 9.00 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6874411, upper bound: 143.6867468
time: 9.18 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -43.5714340, 34.8358727, -40.4853172, 32.3918571, -75.9632721, 75.3211899
1: -35.6103363, 30.9006252, -32.9581909, 28.6525135, -64.2628479, 63.8588181
2: -47.4412842, 31.2789059, -43.8713074, 29.0339985, -76.4752808, 75.1502151
3: -50.7722664, 26.9658852, -47.1991158, 24.9745750, -75.7468414, 74.1650009
4: -47.2406883, 36.3834038, -43.9749947, 33.8210602, -81.0617447, 80.3583984
5: -41.9389191, 32.7521591, -38.9961967, 30.4855728, -72.4244766, 71.7483521
6: -40.4475021, 38.3777542, -37.5641594, 35.6572647, -76.1047668, 75.9418945
7: -43.5059586, 36.6180801, -40.3340111, 33.9742737, -77.4802322, 76.9520874
8: -52.8146362, 36.8984451, -49.0271378, 34.3016510, -87.1162872, 85.9255829
9: -39.7130547, 38.8836021, -36.9770355, 36.0595512, -75.7726059, 75.8606262

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7030985, upper bound: 143.7023283
time: 8.87 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7037381, upper bound: 143.7029257
time: 11.52 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -38.3501244, 30.6847992, -35.8306274, 28.7003212, -67.0504456, 66.5154266
1: -31.3044796, 27.2612934, -29.1249542, 25.4103298, -56.7148094, 56.3862457
2: -41.6513405, 27.5828781, -38.7309303, 25.7499466, -67.4012833, 66.3138046
3: -44.7109795, 23.8439560, -41.7664566, 22.1652317, -66.8761978, 65.6104126
4: -41.6527786, 32.1141129, -39.0092010, 30.0263138, -71.6790924, 71.1233139
5: -36.9308701, 28.8892345, -34.5236130, 27.0189991, -63.9498558, 63.4128456
6: -35.6867943, 33.8188896, -33.3269196, 31.5722027, -67.2589874, 67.1457977
7: -38.3279915, 32.2512169, -35.7002716, 30.0615501, -68.3895416, 67.9514923
8: -46.5639648, 32.5701523, -43.4326019, 30.4615097, -77.0254669, 76.0027542
9: -35.0462837, 34.2864723, -32.7841644, 31.9436722, -66.9899521, 67.0706329

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6873397, upper bound: 143.6866623
time: 10.42 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6870493, upper bound: 143.6864455
time: 8.13 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -41.6860695, 33.3297768, -35.0128479, 28.0611610, -69.7472229, 68.3426208
1: -34.0564003, 29.5947285, -28.4371243, 24.8375721, -58.8939743, 58.0318527
2: -45.3310852, 29.9173412, -37.8170929, 25.1685925, -70.4996796, 67.7344360
3: -48.5969009, 25.8800831, -40.7923889, 21.6739674, -70.2708588, 66.6724701
4: -45.2616653, 34.8486252, -38.1200905, 29.3496838, -74.6113510, 72.9687195
5: -40.1394157, 31.3610382, -33.7340775, 26.3956413, -66.5350342, 65.0951157
6: -38.7357101, 36.7636414, -32.5640373, 30.8434753, -69.5791779, 69.3276672
7: -41.6466179, 35.0256882, -34.8578949, 29.3465939, -70.9932098, 69.8835754
8: -50.5884438, 35.3422050, -42.4413300, 29.8108063, -80.3992462, 77.7835236
9: -38.0565186, 37.2338715, -32.0188751, 31.1896896, -69.2461929, 69.2527466

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6876234, upper bound: 143.6868850
time: 8.32 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6870107, upper bound: 143.6866060
time: 9.37 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -41.6807442, 33.3265266, -40.7987556, 32.6359024, -74.3166428, 74.1252747
1: -33.9534225, 29.4868755, -33.3270302, 28.9602375, -62.9136581, 62.8139038
2: -45.2049942, 29.8753090, -44.3765717, 29.3386745, -74.5436707, 74.2518768
3: -48.6069984, 25.6932392, -47.5342026, 25.2646542, -73.8716507, 73.2274399
4: -45.2669525, 34.8063469, -44.2299232, 34.1086197, -79.3755722, 79.0362701
5: -40.1516266, 31.3860283, -39.2784958, 30.6993484, -70.8509674, 70.6645050
6: -38.6695366, 36.7099571, -37.9091644, 35.9289818, -74.5985107, 74.6191254
7: -41.5498047, 35.0012436, -40.7480965, 34.3217125, -75.8715134, 75.7493439
8: -50.4650269, 35.2543869, -49.4648361, 34.5754585, -85.0404663, 84.7192001
9: -38.0800781, 37.1440315, -37.2076492, 36.4357605, -74.5158310, 74.3516693

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6875134, upper bound: 143.6880362
time: 8.28 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6867463, upper bound: 143.6874413
time: 9.72 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -40.4853172, 32.3918571, -43.5649796, 34.8306732, -75.3159866, 75.9568329
1: -32.9581909, 28.6525135, -35.6051598, 30.8960133, -63.8542023, 64.2576752
2: -43.8713074, 29.0339985, -47.4339790, 31.2744541, -75.1457596, 76.4679794
3: -47.1991158, 24.9745750, -50.7640686, 26.9616146, -74.1607285, 75.7386322
4: -43.9749947, 33.8210602, -47.2335548, 36.3781013, -80.3530960, 81.0546112
5: -38.9961967, 30.4855728, -41.9326630, 32.7472687, -71.7434692, 72.4182053
6: -37.5641594, 35.6572647, -40.4414139, 38.3720360, -75.9361801, 76.0986786
7: -40.3340111, 33.9742737, -43.4989128, 36.6126442, -76.9466553, 77.4731903
8: -49.0271378, 34.3016510, -52.8066864, 36.8931503, -85.9202881, 87.1083374
9: -36.9770355, 36.0595512, -39.7067909, 38.8775139, -75.8545380, 75.7663422

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7023283, upper bound: 143.7030985
time: 8.10 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7029257, upper bound: 143.7037381
time: 10.73 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -35.8306274, 28.7003212, -38.2293930, 30.5875282, -66.4181519, 66.9297180
1: -29.1249542, 25.4103298, -31.2075939, 27.1750546, -56.3000031, 56.6179237
2: -38.7309303, 25.7499466, -41.5144272, 27.4995747, -66.2304993, 67.2643738
3: -41.7664566, 22.1652317, -44.5573463, 23.7639370, -65.5303955, 66.7225723
4: -39.0092010, 30.0263138, -41.5191917, 32.0149689, -71.0241699, 71.5455017
5: -34.5236130, 27.0189991, -36.8137016, 28.7976799, -63.3212891, 63.8326988
6: -33.3269196, 31.5722027, -35.5728416, 33.7117653, -67.0386810, 67.1450348
7: -35.7002716, 30.0615501, -38.1961250, 32.1494370, -67.8497086, 68.2576675
8: -43.4326019, 30.4615097, -46.4151764, 32.4712143, -75.9038162, 76.8766785
9: -32.7841644, 31.9436722, -34.9291725, 34.1723633, -66.9565277, 66.8728485

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6866618, upper bound: 143.6873399
time: 8.22 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6864450, upper bound: 143.6870495
time: 8.77 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -35.0128479, 28.0611610, -41.5588875, 33.2272224, -68.2400665, 69.6200485
1: -28.4371243, 24.8375721, -33.9542923, 29.5038300, -57.9409523, 58.7918625
2: -37.8170929, 25.1685925, -45.1867981, 29.8295403, -67.6466370, 70.3553925
3: -40.7923889, 21.6739674, -48.4350433, 25.7958031, -66.5881805, 70.1090012
4: -38.1200905, 29.3496838, -45.1209297, 34.7441368, -72.8642273, 74.4706116
5: -33.7340775, 26.3956413, -40.0159416, 31.2644901, -64.9985657, 66.4115601
6: -32.5640373, 30.8434753, -38.6156120, 36.6506882, -69.2147141, 69.4590759
7: -34.8578949, 29.3465939, -41.5077095, 34.9184380, -69.7763214, 70.8543015
8: -42.4413300, 29.8108063, -50.4316177, 35.2378998, -77.6792221, 80.2424240
9: -32.0188751, 31.1896896, -37.9331169, 37.1136131, -69.1324921, 69.1228027

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6868850, upper bound: 143.6876234
time: 8.40 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6866060, upper bound: 143.6872658
time: 7.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -38.3632622, 30.6995811, -37.6716499, 30.1500988, -68.5133591, 68.3712311
1: -31.2099094, 27.1721115, -30.6320877, 26.6860142, -57.8959236, 57.8041992
2: -41.5368958, 27.5338326, -40.7691154, 27.0477867, -68.5846863, 68.3029327
3: -44.7214317, 23.7026501, -43.9347267, 23.2563591, -67.9777832, 67.6373596
4: -41.7076836, 32.0954132, -40.9853020, 31.5200310, -73.2277145, 73.0807190
5: -36.9511223, 28.9016132, -36.3165894, 28.3930855, -65.3441925, 65.2182007
6: -35.6323051, 33.7963600, -35.0350838, 33.1729164, -68.8052216, 68.8314438
7: -38.2335587, 32.1949501, -37.5481987, 31.6264172, -69.8599701, 69.7431412
8: -46.4747887, 32.5359039, -45.6107903, 31.9182549, -78.3930359, 78.1466904
9: -35.0732269, 34.2001190, -34.4375420, 33.5443039, -68.6175308, 68.6376572

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6860769, upper bound: 143.6857658
time: 7.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6858052, upper bound: 143.6856097
time: 7.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -37.1602249, 29.7646084, -39.8990021, 31.9125538, -69.0727768, 69.6636047
1: -30.2116051, 26.3339100, -32.4541626, 28.2411060, -58.4527130, 58.7880707
2: -40.2023811, 26.6862850, -43.2111969, 28.5949478, -68.7973251, 69.8974762
3: -43.3046379, 22.9807472, -46.5278091, 24.6197433, -67.9243774, 69.5085449
4: -40.4122238, 31.1067448, -43.4087029, 33.3409538, -73.7531738, 74.5154266
5: -35.7919922, 27.9980869, -38.4481316, 30.0417118, -65.8337021, 66.4462204
6: -34.5229912, 32.7386513, -37.0690804, 35.1429100, -69.6659012, 69.8077316
7: -37.0125618, 31.1614456, -39.7643738, 33.4627609, -70.4753265, 70.9258194
8: -45.0331039, 31.5821838, -48.2989731, 33.7798347, -78.8129425, 79.8811493
9: -33.9668312, 33.1119804, -36.4397697, 35.5018196, -69.4686508, 69.5517502

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6864380, upper bound: 143.6863306
time: 7.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6858405, upper bound: 143.6855859
time: 9.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -33.2130470, 26.6988087, -30.7308960, 24.7645359, -57.9775848, 57.4297028
1: -26.9820557, 23.6190014, -24.9307098, 21.8856907, -48.8677406, 48.5497131
2: -35.8603210, 23.8844643, -33.1453934, 22.1680603, -58.0283813, 57.0298576
3: -38.6620636, 20.6457138, -35.7316704, 19.0673561, -57.7294159, 56.3773842
4: -36.1943359, 27.9230728, -33.5142860, 25.9103642, -62.1046982, 61.4373589
5: -32.0040054, 25.1070766, -29.6369419, 23.2569504, -55.2609482, 54.7440186
6: -30.8493309, 29.3316040, -28.5828915, 27.1216717, -57.9710007, 57.9144821
7: -33.0531197, 27.8220425, -30.5505753, 25.7514362, -58.8045540, 58.3726196
8: -40.3354836, 28.4179974, -37.3074455, 26.3911457, -66.7266006, 65.7254333
9: -30.4175873, 29.6813583, -28.1294327, 27.4372444, -57.8548279, 57.8107872

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6852384, upper bound: 143.6853348
time: 7.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6851824, upper bound: 143.6853495
time: 6.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -30.0866299, 24.2319317, -29.2421837, 23.5756397, -53.6622696, 53.4741135
1: -24.3734760, 21.4270458, -23.6693802, 20.8331261, -45.2065964, 45.0964241
2: -32.4071426, 21.6747780, -31.4827404, 21.1056881, -53.5128326, 53.1575165
3: -34.9363670, 18.7622700, -33.9169655, 18.1839962, -53.1203613, 52.6792336
4: -32.8359947, 25.3489017, -31.9208431, 24.6609974, -57.4969940, 57.2697449
5: -29.0058956, 22.7866249, -28.2092628, 22.1532974, -51.1591949, 50.9958878
6: -27.9930897, 26.5850086, -27.2266350, 25.8124180, -53.8055077, 53.8116455
7: -29.9161301, 25.1547413, -29.0472755, 24.4528713, -54.3690033, 54.2020111
8: -36.5375938, 25.8407516, -35.4752960, 25.1522636, -61.6898537, 61.3160477
9: -27.5528889, 26.8621368, -26.7462826, 26.0679665, -53.6208572, 53.6084213

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7005587, upper bound: 143.7005122
time: 6.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7016060, upper bound: 143.7016060
time: 6.11 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.85 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6903316, upper bound: 143.6910758
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7281920, upper bound: 143.7280055
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6911738, upper bound: 143.6901291
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7292412, upper bound: 143.7290917
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6898251, upper bound: 143.6907170
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7278172, upper bound: 143.7277790
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6899517, upper bound: 143.6909324
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7288723, upper bound: 143.7288723
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6880362, upper bound: 143.6875134
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6874411, upper bound: 143.6867468
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7030985, upper bound: 143.7023283
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7037381, upper bound: 143.7029257
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6873397, upper bound: 143.6866623
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6870493, upper bound: 143.6864455
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6876234, upper bound: 143.6868850
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6870107, upper bound: 143.6866060
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6875134, upper bound: 143.6880362
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6867463, upper bound: 143.6874413
IS_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7023283, upper bound: 143.7030985
IS_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7029257, upper bound: 143.7037381
IS_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6866618, upper bound: 143.6873399
IS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6864450, upper bound: 143.6870495
IS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6868850, upper bound: 143.6876234
IS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6866060, upper bound: 143.6872658
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6860769, upper bound: 143.6857658
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6858052, upper bound: 143.6856097
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6864380, upper bound: 143.6863306
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6858405, upper bound: 143.6855859
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6852384, upper bound: 143.6853348
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.6851824, upper bound: 143.6853495
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7005587, upper bound: 143.7005122
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 13.85
Output dim: 4, lower bound: -143.7016060, upper bound: 143.7016060

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -27.3466244, 22.0212688, -30.2669392, 24.3350182, -51.6816406, 52.2882080
1: -22.0880089, 19.5258961, -24.5444260, 21.5809536, -43.6689606, 44.0703201
2: -29.3784847, 19.8369751, -32.6516800, 21.9189663, -51.2974434, 52.4886551
3: -31.6967010, 17.1019192, -35.1934738, 18.8535080, -50.5502014, 52.2953949
4: -29.8276081, 22.9460125, -32.9848442, 25.3835545, -55.2111626, 55.9308548
5: -26.3785553, 20.5970840, -29.2123280, 22.7941437, -49.1726990, 49.8094063
6: -25.5856819, 24.0455189, -28.3065586, 26.6248608, -52.2105408, 52.3520775
7: -27.1690407, 22.8171272, -30.1345253, 25.3684826, -52.5375214, 52.9516525
8: -33.1881905, 23.5270081, -36.7431107, 25.9068394, -59.0950317, 60.2701149
9: -24.8637028, 24.2452908, -27.5619164, 26.8874092, -51.7511139, 51.8072052

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6893507, upper bound: 143.6901456
time: 10.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6903321, upper bound: 143.6910758
time: 8.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 20.84 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 20.84
Output dim: 4, lower bound: -143.6893507, upper bound: 143.6901456
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 20.84
Output dim: 4, lower bound: -143.6903321, upper bound: 143.6910758
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7281920, upper bound: 143.7280055
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6911738, upper bound: 143.6901291
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7292412, upper bound: 143.7290917
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6898251, upper bound: 143.6907170
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7278172, upper bound: 143.7277790
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6899517, upper bound: 143.6909324
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7288723, upper bound: 143.7288723
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6880362, upper bound: 143.6875134
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6874411, upper bound: 143.6867468
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7030985, upper bound: 143.7023283
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7037381, upper bound: 143.7029257
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6873397, upper bound: 143.6866623
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6870493, upper bound: 143.6864455
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6876234, upper bound: 143.6868850
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6870107, upper bound: 143.6866060
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6875134, upper bound: 143.6880362
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6867463, upper bound: 143.6874413
IS_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7023283, upper bound: 143.7030985
IS_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7029257, upper bound: 143.7037381
IS_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6866618, upper bound: 143.6873399
IS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6864450, upper bound: 143.6870495
IS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6868850, upper bound: 143.6876234
IS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6866060, upper bound: 143.6872658
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6860769, upper bound: 143.6857658
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6858052, upper bound: 143.6856097
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6864380, upper bound: 143.6863306
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6858405, upper bound: 143.6855859
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6852384, upper bound: 143.6853348
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.6851824, upper bound: 143.6853495
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7005587, upper bound: 143.7005122
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.84
Output dim: 4, lower bound: -143.7016060, upper bound: 143.7016060
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1827.77 seconds
