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
execution time: IAR + LP analysis = 1.19 + 3.92 = 5.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -7.1077352, upper bound: 7.1077352


# Binary Search by BASE starts (time budget: 2694.90 seconds, max iter: 100)

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
Binary search time: 22.68 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2672.22 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
time: 2.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278448
time: 1.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.98 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.98
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.98
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278448

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.9998550, 2.8064208, -4.0230141, 2.8226054, -6.8224602, 6.8294349
1: -2.7389779, 3.0692990, -2.7556953, 3.0856514, -5.8246293, 5.8249941
2: -3.9650166, 3.0592949, -3.9878938, 3.0745039, -7.0395203, 7.0471888
3: -4.7972660, 2.4239097, -4.8247290, 2.4369347, -7.2342005, 7.2486386
4: -5.0186734, 3.1929450, -5.0468745, 3.2078700, -8.2265434, 8.2398195
5: -4.1977353, 2.6268024, -4.2217145, 2.6404290, -6.8381643, 6.8485169
6: -4.8564968, 2.9363008, -4.8847556, 2.9497087, -7.8062057, 7.8210564
7: -3.6791067, 3.6800561, -3.6985793, 3.6995137, -7.3786201, 7.3786354
8: -5.3407345, 2.7827613, -5.3712187, 2.7974010, -8.1381359, 8.1539803
9: -3.5578074, 3.5852487, -3.5766788, 3.6042769, -7.1620846, 7.1619272

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5044865
time: 2.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
time: 2.15 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.9337635, 5.4761028, -3.9896588, 2.7992415, -10.7330055, 9.4657612
1: -5.6388774, 5.8590884, -2.7316647, 3.0620725, -8.7009497, 8.5907536
2: -7.8929882, 5.6129866, -3.9549851, 3.0525610, -10.9455490, 9.5679722
3: -9.5662737, 4.5686665, -4.7852449, 2.4181046, -11.9843788, 9.3539114
4: -9.9575558, 5.6808515, -5.0063491, 3.1863890, -13.1439447, 10.6872005
5: -8.3837214, 4.9044352, -4.1872072, 2.6207905, -11.0045118, 9.0916424
6: -9.7087727, 4.9445033, -4.8440862, 2.9303930, -12.6391659, 9.7885895
7: -7.0684695, 7.1211610, -3.6705933, 3.6715269, -10.7399960, 10.7917538
8: -10.5496216, 4.9511232, -5.3273201, 2.7761736, -13.3257952, 10.2784433
9: -6.8049579, 6.9037161, -3.5494895, 3.5769305, -10.3818884, 10.4532051

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5206759, upper bound: 6.4883889
time: 1.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278448
time: 1.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 9.96
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5044865
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.96
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 9.96
Output dim: 6, lower bound: -6.5206759, upper bound: 6.4883889
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 9.96
Output dim: 6, lower bound: -6.5278448, upper bound: 6.5278448

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -3.8669295, 2.7138901, -2.0116720, 1.4323316, -5.2992611, 4.7255621
1: -2.6415226, 2.9743705, -1.4668773, 1.5724030, -4.2139254, 4.4412479
2: -3.8329148, 2.9718578, -1.9007626, 1.7571841, -5.5900989, 4.8726206
3: -4.6378870, 2.3511825, -2.3158488, 1.2986166, -5.9365034, 4.6670313
4: -4.8530397, 3.1062758, -2.3877773, 1.8443369, -6.6973767, 5.4940529
5: -4.0571804, 2.5483727, -2.0670991, 1.4813883, -5.5385685, 4.6154718
6: -4.6907411, 2.8614917, -2.2512043, 1.8416957, -6.5324368, 5.1126957
7: -3.5657084, 3.5657601, -1.9117014, 1.9263825, -5.4920912, 5.4774618
8: -5.1631427, 2.7009425, -2.5493827, 1.5958726, -6.7590151, 5.2503252
9: -3.4484551, 3.4743092, -1.8408734, 1.8823425, -5.3307977, 5.3151827

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5044865
time: 4.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5044865
time: 3.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.9707305, 2.7861738, -3.4197326, 2.4042311, -6.3749619, 6.2059064
1: -2.7176411, 3.0485206, -2.3395681, 2.6527512, -5.3703923, 5.3880887
2: -3.9360859, 3.0401549, -3.3855290, 2.6847005, -6.6207867, 6.4256840
3: -4.7622862, 2.4079762, -4.1023693, 2.1066570, -6.8689432, 6.5103455
4: -4.9824133, 3.1739476, -4.2932830, 2.8134782, -7.7958918, 7.4672308
5: -4.1669755, 2.6096222, -3.5826735, 2.2871852, -6.4541607, 6.1922960
6: -4.8202038, 2.9199367, -4.1311321, 2.6151648, -7.4353685, 7.0510688
7: -3.6542761, 3.6550233, -3.1846676, 3.1828761, -6.8371525, 6.8396912
8: -5.3018527, 2.7648606, -4.5642457, 2.4273038, -7.7291565, 7.3291063
9: -3.5338497, 3.5609365, -3.0789995, 3.1036978, -6.6375475, 6.6399360

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
time: 2.65 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
time: 2.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.7969294, 5.3811522, -1.9826281, 1.4147588, -9.2116880, 7.3637800
1: -5.5386343, 5.7612519, -1.4489744, 1.5506973, -7.0893316, 7.2102261
2: -7.7561955, 5.5230923, -1.8706522, 1.7386198, -9.4948158, 7.3937445
3: -9.4014435, 4.4939227, -2.2802219, 1.2818022, -10.6832457, 6.7741446
4: -9.7867956, 5.5919924, -2.3534858, 1.8246503, -11.6114464, 7.9454784
5: -8.2390566, 4.8234558, -2.0386076, 1.4654629, -9.7045193, 6.8620634
6: -9.5374289, 4.8678951, -2.2131865, 1.8271095, -11.3645382, 7.0810814
7: -6.9512539, 7.0031185, -1.8863658, 1.9017102, -8.8529644, 8.8894844
8: -10.3661880, 4.8683138, -2.5089645, 1.5774460, -11.9436340, 7.3772783
9: -6.6924982, 6.7882519, -1.8160733, 1.8589506, -8.5514488, 8.6043253

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4873017, upper bound: 6.4873017
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4873017, upper bound: 6.4883889
time: 1.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.9047427, 5.4559975, -3.3878686, 2.3820851, -10.2868280, 8.8438663
1: -5.6176171, 5.8383694, -2.3208337, 2.6298237, -8.2474403, 8.1592026
2: -7.8639784, 5.5939493, -3.3537014, 2.6646121, -10.5285902, 8.9476509
3: -9.5312920, 4.5528445, -4.0647874, 2.0886979, -11.6199894, 8.6176319
4: -9.9213295, 5.6619821, -4.2540646, 2.7927587, -12.7140884, 9.9160461
5: -8.3530579, 4.8872747, -3.5494554, 2.2686324, -10.6216908, 8.4367304
6: -9.6724043, 4.9282303, -4.0918818, 2.5975196, -12.2699242, 9.0201120
7: -7.0436273, 7.0961709, -3.1579504, 3.1564798, -10.2001076, 10.2541218
8: -10.5107222, 4.9335604, -4.5220704, 2.4070859, -12.9178085, 9.4556313
9: -6.7811227, 6.8792319, -3.0526905, 3.0781517, -9.8592739, 9.9319229

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4883889, upper bound: 6.5206759
time: 1.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4883889, upper bound: 6.5278448
time: 1.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.54 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5044865
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5044865
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.9567969, upper bound: 6.5435228
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.4873017, upper bound: 6.4873017
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.4873017, upper bound: 6.4883889
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.4883889, upper bound: 6.5206759
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 6, lower bound: -6.4883889, upper bound: 6.5278448

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.9907990, 1.4196956, -2.0116720, 1.4323316, -3.4231305, 3.4313676
1: -1.4539746, 1.5567937, -1.4668773, 1.5724030, -3.0263777, 3.0236712
2: -1.8790685, 1.7438605, -1.9007626, 1.7571841, -3.6362526, 3.6446230
3: -2.2901187, 1.2865995, -2.3158488, 1.2986166, -3.5887353, 3.6024485
4: -2.3630099, 1.8301512, -2.3877773, 1.8443369, -4.2073469, 4.2179284
5: -2.0465465, 1.4699534, -2.0670991, 1.4813883, -3.5279348, 3.5370526
6: -2.2238126, 1.8311768, -2.2512043, 1.8416957, -4.0655084, 4.0823812
7: -1.8934288, 1.9085836, -1.9117014, 1.9263825, -3.8198113, 3.8202851
8: -2.5202506, 1.5827625, -2.5493827, 1.5958726, -4.1161232, 4.1321449
9: -1.8230120, 1.8654860, -1.8408734, 1.8823425, -3.7053545, 3.7063594

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3526636, upper bound: 6.3367574
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8000287, upper bound: 6.3258261
time: 2.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8925563, upper bound: 6.4596390
time: 2.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.3972940, 2.3886709, -2.0116720, 1.4323316, -4.8296256, 4.4003429
1: -2.3263485, 2.6366069, -1.4668773, 1.5724030, -3.8987515, 4.1034842
2: -3.3630967, 2.6705737, -1.9007626, 1.7571841, -5.1202807, 4.5713363
3: -4.0758300, 2.0940685, -2.3158488, 1.2986166, -5.3744469, 4.4099174
4: -4.2655725, 2.7988687, -2.3877773, 1.8443369, -6.1099095, 5.1866460
5: -3.5592384, 2.2741256, -2.0670991, 1.4813883, -5.0406265, 4.3412247
6: -4.1034451, 2.6027677, -2.2512043, 1.8416957, -5.9451408, 4.8539720
7: -3.1658087, 3.1642411, -1.9117014, 1.9263825, -5.0921912, 5.0759425
8: -4.5345249, 2.4131799, -2.5493827, 1.5958726, -6.1303978, 4.9625626
9: -3.0604610, 3.0856752, -1.8408734, 1.8823425, -4.9428034, 4.9265485

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3526636, upper bound: 6.3367574
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4572687
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8925563, upper bound: 6.4596387
time: 3.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.9907990, 1.4196956, -3.4197326, 2.4042311, -4.3950300, 4.8394279
1: -1.4539746, 1.5567937, -2.3395681, 2.6527512, -4.1067257, 3.8963618
2: -1.8790685, 1.7438605, -3.3855290, 2.6847005, -4.5637689, 5.1293898
3: -2.2901187, 1.2865995, -4.1023693, 2.1066570, -4.3967757, 5.3889689
4: -2.3630099, 1.8301512, -4.2932830, 2.8134782, -5.1764879, 6.1234341
5: -2.0465465, 1.4699534, -3.5826735, 2.2871852, -4.3337317, 5.0526271
6: -2.2238126, 1.8311768, -4.1311321, 2.6151648, -4.8389773, 5.9623089
7: -1.8934288, 1.9085836, -3.1846676, 3.1828761, -5.0763049, 5.0932512
8: -2.5202506, 1.5827625, -4.5642457, 2.4273038, -4.9475546, 6.1470079
9: -1.8230120, 1.8654860, -3.0789995, 3.1036978, -4.9267097, 4.9444857

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3526636, upper bound: 6.3962403
time: 1.67 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8000287, upper bound: 6.3934877
time: 2.60 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8925563, upper bound: 6.4986144
time: 1.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.3972940, 2.3886709, -3.4197326, 2.4042311, -5.8015251, 5.8084035
1: -2.3263485, 2.6366069, -2.3395681, 2.6527512, -4.9790998, 4.9761753
2: -3.3630967, 2.6705737, -3.3855290, 2.6847005, -6.0477972, 6.0561028
3: -4.0758300, 2.0940685, -4.1023693, 2.1066570, -6.1824870, 6.1964378
4: -4.2655725, 2.7988687, -4.2932830, 2.8134782, -7.0790510, 7.0921516
5: -3.5592384, 2.2741256, -3.5826735, 2.2871852, -5.8464236, 5.8567991
6: -4.1034451, 2.6027677, -4.1311321, 2.6151648, -6.7186098, 6.7339001
7: -3.1658087, 3.1642411, -3.1846676, 3.1828761, -6.3486848, 6.3489084
8: -4.5345249, 2.4131799, -4.5642457, 2.4273038, -6.9618287, 6.9774256
9: -3.0604610, 3.0856752, -3.0789995, 3.1036978, -6.1641588, 6.1646748

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3526636, upper bound: 6.3835894
time: 2.67 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9229722, upper bound: 6.5424637
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5432528
time: 2.62 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.4994006, 3.7930660, -1.9826281, 1.4147588, -6.9141593, 5.7756939
1: -3.8520403, 4.1162167, -1.4489744, 1.5506973, -5.4027376, 5.5651913
2: -5.4571419, 4.0187263, -1.8706522, 1.7386198, -7.1957617, 5.8893785
3: -6.6383553, 3.2404628, -2.2802219, 1.2818022, -7.9201574, 5.5206847
4: -6.9012947, 4.0979290, -2.3534858, 1.8246503, -8.7259445, 6.4514151
5: -5.8079453, 3.4673388, -2.0386076, 1.4654629, -7.2734079, 5.5059462
6: -6.6508002, 3.5952442, -2.2131865, 1.8271095, -8.4779100, 5.8084307
7: -4.9809837, 5.0101261, -1.8863658, 1.9017102, -6.8826938, 6.8964920
8: -7.2896943, 3.5155940, -2.5089645, 1.5774460, -8.8671398, 6.0245585
9: -4.8012891, 4.8493805, -1.8160733, 1.8589506, -6.6602397, 6.6654539

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3040615
time: 1.75 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4409483
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3151188, 5.0462313, -1.9826281, 1.4147588, -8.7298775, 7.0288591
1: -5.1856198, 5.4158859, -1.4489744, 1.5506973, -6.7363172, 6.8648605
2: -7.2729502, 5.2057147, -1.8706522, 1.7386198, -9.0115700, 7.0763669
3: -8.8203754, 4.2304759, -2.2802219, 1.2818022, -10.1021776, 6.5106978
4: -9.1864567, 5.2778506, -2.3534858, 1.8246503, -11.0111065, 7.6313362
5: -7.7295513, 4.5377345, -2.0386076, 1.4654629, -9.1950140, 6.5763421
6: -8.9329529, 4.5965724, -2.2131865, 1.8271095, -10.7600622, 6.8097591
7: -6.5381107, 6.5868416, -1.8863658, 1.9017102, -8.4398212, 8.4732075
8: -9.7187939, 4.5761771, -2.5089645, 1.5774460, -11.2962399, 7.0851417
9: -6.2953672, 6.3807459, -1.8160733, 1.8589506, -8.1543179, 8.1968193

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3056622
time: 2.25 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4421332
time: 1.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.4994006, 3.7930660, -3.3878686, 2.3820851, -7.8814859, 7.1809349
1: -3.8520403, 4.1162167, -2.3208337, 2.6298237, -6.4818640, 6.4370503
2: -5.4571419, 4.0187263, -3.3537014, 2.6646121, -8.1217537, 7.3724279
3: -6.6383553, 3.2404628, -4.0647874, 2.0886979, -8.7270527, 7.3052502
4: -6.9012947, 4.0979290, -4.2540646, 2.7927587, -9.6940536, 8.3519936
5: -5.8079453, 3.4673388, -3.5494554, 2.2686324, -8.0765781, 7.0167942
6: -6.6508002, 3.5952442, -4.0918818, 2.5975196, -9.2483196, 7.6871262
7: -4.9809837, 5.0101261, -3.1579504, 3.1564798, -8.1374636, 8.1680765
8: -7.2896943, 3.5155940, -4.5220704, 2.4070859, -9.6967802, 8.0376644
9: -4.8012891, 4.8493805, -3.0526905, 3.0781517, -7.8794408, 7.9020710

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3040615
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4742714
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3151188, 5.0462313, -3.3878686, 2.3820851, -9.6972036, 8.4341002
1: -5.1856198, 5.4158859, -2.3208337, 2.6298237, -7.8154435, 7.7367196
2: -7.2729502, 5.2057147, -3.3537014, 2.6646121, -9.9375620, 8.5594158
3: -8.8203754, 4.2304759, -4.0647874, 2.0886979, -10.9090729, 8.2952633
4: -9.1864567, 5.2778506, -4.2540646, 2.7927587, -11.9792156, 9.5319157
5: -7.7295513, 4.5377345, -3.5494554, 2.2686324, -9.9981842, 8.0871897
6: -8.9329529, 4.5965724, -4.0918818, 2.5975196, -11.5304728, 8.6884537
7: -6.5381107, 6.5868416, -3.1579504, 3.1564798, -9.6945906, 9.7447920
8: -9.7187939, 4.5761771, -4.5220704, 2.4070859, -12.1258793, 9.0982475
9: -6.2953672, 6.3807459, -3.0526905, 3.0781517, -9.3735189, 9.4334364

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3745980
time: 1.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4816400
time: 1.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 12.50 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.8000287, upper bound: 6.3258261
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.8925563, upper bound: 6.4596390
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4572687
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.8925563, upper bound: 6.4596387
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.8000287, upper bound: 6.3934877
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.8925563, upper bound: 6.4986144
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.9229722, upper bound: 6.5424637
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.9328152, upper bound: 6.5432528
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3040615
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4409483
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3056622
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4421332
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3040615
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4742714
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.3815016, upper bound: 6.3745980
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.50
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4816400

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6656407, 1.2240677, -0.5120767, 0.5691385, -2.2347794, 1.7361444
1: -1.2571090, 1.3049489, -0.5127921, 0.5291355, -1.7862445, 1.8177410
2: -1.5417174, 1.5295925, -0.4862878, 0.7225779, -2.2447472, 2.0158803
3: -1.8809102, 1.1080421, -0.5061171, 0.4541504, -2.3350606, 1.6141592
4: -1.9754969, 1.6019256, -0.6220254, 0.7045293, -2.6800263, 2.2239511
5: -1.7189161, 1.2857982, -0.6345273, 0.6183742, -2.3372903, 1.9203255
6: -1.7605348, 1.6693459, -0.0485611, 1.3042519, -3.0647867, 1.7179070
7: -1.6105235, 1.6304141, -0.6513570, 0.6349446, -2.2454681, 2.2817712
8: -2.0523453, 1.3436480, -0.6258660, 0.6162184, -2.6685638, 1.9695139
9: -1.5437493, 1.5903418, -0.5640154, 0.6615943, -2.2053437, 2.1543572

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7505200, upper bound: 6.3247220
time: 2.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7361955, upper bound: 6.2828303
time: 3.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.9907990, 1.4196956, -1.1420355, 0.9326021, -2.9234011, 2.5617311
1: -1.4539746, 1.5567937, -0.9484642, 0.9121683, -2.3661427, 2.5052578
2: -1.8790685, 1.7438605, -1.0203673, 1.1916219, -3.0706904, 2.7642279
3: -2.2901187, 1.2865995, -1.2077805, 0.8196509, -3.1097696, 2.4943800
4: -2.3630099, 1.8301512, -1.3435372, 1.2190652, -3.5820751, 3.1736884
5: -2.0465465, 1.4699534, -1.1887773, 1.0055324, -3.0520787, 2.6587307
6: -2.2238126, 1.8311768, -0.9710069, 1.4670855, -3.6908979, 2.8021836
7: -1.8934288, 1.9085836, -1.1517854, 1.1832545, -3.0766833, 3.0603690
8: -2.5202506, 1.5827625, -1.3827407, 0.9887957, -3.5090463, 2.9655032
9: -1.8230120, 1.8654860, -1.0900233, 1.1778665, -3.0008783, 2.9555092

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7453333, upper bound: 6.4518258
time: 2.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4572687
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4596387
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.4452050, 1.0976579, -1.6845766, 1.2350743, -2.6802793, 2.7822347
1: -1.1187904, 1.1319900, -1.2686191, 1.3191504, -2.4379408, 2.4006090
2: -1.3064896, 1.3801873, -1.5612864, 1.5419313, -2.8484209, 2.9414737
3: -1.6011243, 0.9846236, -1.9055444, 1.1183335, -2.7194576, 2.8901680
4: -1.7160301, 1.3992213, -1.9989440, 1.6146653, -3.3306954, 3.3981652
5: -1.4969268, 1.1538793, -1.7382562, 1.2958132, -2.7927399, 2.8921356
6: -1.3699579, 1.5579607, -1.7867347, 1.6773891, -3.0473471, 3.3446956
7: -1.4039067, 1.4422271, -1.6272955, 1.6465812, -3.0504880, 3.0695226
8: -1.7664710, 1.1283426, -2.0795779, 1.3546505, -3.1211214, 3.2079206
9: -1.3421528, 1.4131497, -1.5601279, 1.6063154, -2.9484682, 2.9732776

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8056643, upper bound: 6.4557471
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4572687
time: 2.50 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -2.4352596, 1.6902825, -2.0116720, 1.4323316, -3.8675911, 3.7019544
1: -1.7299993, 1.8822000, -1.4668773, 1.5724030, -3.3024023, 3.3490772
2: -2.3419652, 2.0217772, -1.9007626, 1.7571841, -4.0991492, 3.9225397
3: -2.8765233, 1.5369530, -2.3158488, 1.2986166, -4.1751399, 3.8528018
4: -2.9759130, 2.1076705, -2.3877773, 1.8443369, -4.8202500, 4.4954481
5: -2.5137696, 1.6964197, -2.0670991, 1.4813883, -3.9951580, 3.7635188
6: -2.7755597, 2.0208044, -2.2512043, 1.8416957, -4.6172552, 4.2720089
7: -2.3096671, 2.3147614, -1.9117014, 1.9263825, -4.2360497, 4.2264628
8: -3.1435366, 1.7078052, -2.5493827, 1.5958726, -4.7394094, 4.2571878
9: -2.2158926, 2.2426958, -1.8408734, 1.8823425, -4.0982351, 4.0835690

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3030238, upper bound: 6.2870188
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8000287, upper bound: 6.3258261
time: 2.26 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8000287, upper bound: 6.4596387
time: 2.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6656407, 1.2240677, -1.4627786, 1.1071156, -2.7727563, 2.6868463
1: -1.2571090, 1.3049489, -1.1290582, 1.1447830, -2.4018922, 2.4340072
2: -1.5417174, 1.5295925, -1.3234627, 1.3910390, -2.9327564, 2.8530552
3: -1.8809102, 1.1080421, -1.6237936, 0.9941247, -2.8750348, 2.7318358
4: -1.9754969, 1.6019256, -1.7374651, 1.4111998, -3.3866968, 3.3393908
5: -1.7189161, 1.2857982, -1.5144646, 1.1628919, -2.8818078, 2.8002629
6: -1.7605348, 1.6693459, -1.3942568, 1.5648355, -3.3253703, 3.0636027
7: -1.6105235, 1.6304141, -1.4195868, 1.4568799, -3.0674033, 3.0500009
8: -2.0523453, 1.3436480, -1.7888521, 1.1374334, -3.1897788, 3.1325002
9: -1.5437493, 1.5903418, -1.3572760, 1.4267188, -2.9704680, 2.9476178

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6945700, upper bound: 6.3846851
time: 2.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8371029, upper bound: 6.3921976
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8540009, upper bound: 6.3934877
time: 2.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.9907990, 1.4196956, -2.4546261, 1.7031093, -3.6939082, 3.8743217
1: -1.4539746, 1.5567937, -1.7421445, 1.8970578, -3.3510323, 3.2989383
2: -1.8790685, 1.7438605, -2.3626561, 2.0345712, -3.9136395, 4.1065168
3: -2.2901187, 1.2865995, -2.9013157, 1.5480387, -3.8381574, 4.1879153
4: -2.3630099, 1.8301512, -3.0025525, 2.1210904, -4.4841003, 4.8327036
5: -2.0465465, 1.4699534, -2.5337820, 1.7074968, -3.7540431, 4.0037355
6: -2.2238126, 1.8311768, -2.8016455, 2.0314617, -4.2552743, 4.6328220
7: -1.8934288, 1.9085836, -2.3273025, 2.3318105, -4.2252393, 4.2358861
8: -2.5202506, 1.5827625, -3.1717901, 1.7196927, -4.2399435, 4.7545528
9: -1.8230120, 1.8654860, -2.2328882, 2.2598791, -4.0828910, 4.0983744

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3972508, upper bound: 6.3476398
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8534154, upper bound: 6.4965538
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8534154, upper bound: 6.4986144
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -2.5346899, 1.7765908, -3.2318163, 2.2707880, -4.8054781, 5.0084071
1: -1.7901181, 1.9801139, -2.2242961, 2.5123715, -4.3024898, 4.2044101
2: -2.4560161, 2.1001408, -3.1910393, 2.5625408, -5.0185566, 5.2911801
3: -2.9814534, 1.5958202, -3.8683987, 2.0008070, -4.9822607, 5.4642191
4: -3.1020710, 2.2147961, -4.0460958, 2.6857333, -5.7878046, 6.2608919
5: -2.6071324, 1.7844950, -3.3758826, 2.1791232, -4.7862558, 5.1603775
6: -2.9721599, 2.1324494, -3.8872409, 2.5111716, -5.4833317, 6.0196905
7: -2.3860393, 2.3968279, -3.0183990, 3.0184872, -5.4045267, 5.4152269
8: -3.3103840, 1.9042680, -4.3032899, 2.3143644, -5.6247482, 6.2075577
9: -2.3030319, 2.3400958, -2.9171708, 2.9446435, -5.2476754, 5.2572665

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8240702, upper bound: 6.3919439
time: 2.48 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8952090, upper bound: 6.4975010
time: 2.08 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -2.9839191, 2.0893974, -3.4197326, 2.4042311, -5.3881502, 5.5091300
1: -2.0696969, 2.3223333, -2.3395681, 2.6527512, -4.7224483, 4.6619015
2: -2.9296746, 2.3971224, -3.3855290, 2.6847005, -5.6143751, 5.7826514
3: -3.5546443, 1.8568648, -4.1023693, 2.1066570, -5.6613016, 5.9592342
4: -3.7117026, 2.5152025, -4.2932830, 2.8134782, -6.5251808, 6.8084855
5: -3.0967796, 2.0373113, -3.5826735, 2.2871852, -5.3839645, 5.6199846
6: -3.5582290, 2.3696117, -4.1311321, 2.6151648, -6.1733937, 6.5007439
7: -2.7946949, 2.7967134, -3.1846676, 3.1828761, -5.9775710, 5.9813809
8: -3.9501107, 2.1604552, -4.5642457, 2.4273038, -6.3774147, 6.7247009
9: -2.6985760, 2.7296758, -3.0789995, 3.1036978, -5.8022738, 5.8086753

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4197016, upper bound: 6.3835894
time: 2.91 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8394065, upper bound: 6.3931464
time: 1.74 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9037576, upper bound: 6.4983196
time: 2.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.1658010, 3.5473330, -0.5005908, 0.5617379, -5.7275391, 4.0479240
1: -3.6093550, 3.8679023, -0.5036243, 0.5221089, -4.1314640, 4.3715267
2: -5.1136460, 3.7963769, -0.4780582, 0.7136130, -5.8272591, 4.2744350
3: -6.2369308, 3.0580466, -0.4969927, 0.4466695, -6.6836004, 3.5550394
4: -6.4826913, 3.8679109, -0.6103132, 0.6945386, -7.1772299, 4.4782243
5: -5.4552679, 3.2604082, -0.6252642, 0.6109307, -6.0661983, 3.8856723
6: -6.2125249, 3.3894362, -0.0333656, 1.3015839, -7.5141087, 3.4228020
7: -4.6971464, 4.7217884, -0.6434321, 0.6239196, -5.3210659, 5.3652205
8: -6.8303909, 3.2712245, -0.6138540, 0.6095831, -7.4399738, 3.8850784
9: -4.5199256, 4.5636482, -0.5549438, 0.6503330, -5.1702585, 5.1185923

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3588861, upper bound: 6.2609699
time: 1.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3360330, upper bound: 6.2600941
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.4994006, 3.7930660, -1.1182098, 0.9202838, -6.4196844, 4.9112759
1: -3.8520403, 4.1162167, -0.9345390, 0.8959125, -4.7479529, 5.0507555
2: -5.4571419, 4.0187263, -0.9988400, 1.1763062, -6.6334481, 5.0175662
3: -6.6383553, 3.2404628, -1.1769482, 0.8068380, -7.4451933, 4.4174109
4: -6.9012947, 4.0979290, -1.3143544, 1.2029325, -8.1042271, 5.4122834
5: -5.8079453, 3.4673388, -1.1651046, 0.9937537, -6.8016992, 4.6324434
6: -6.6508002, 3.5952442, -0.9370954, 1.4597459, -8.1105461, 4.5323396
7: -4.9809837, 5.0101261, -1.1328228, 1.1643587, -6.1453424, 6.1429491
8: -7.2896943, 3.5155940, -1.3536148, 0.9758874, -8.2655821, 4.8692088
9: -4.8012891, 4.8493805, -1.0703744, 1.1594752, -5.9607644, 5.9197550

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4002567, upper bound: 6.4358400
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3993058, upper bound: 6.3993058
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.9803391, 4.8028984, -0.5005908, 0.5617379, -7.5420771, 5.3034892
1: -4.9428368, 5.1700234, -0.5036243, 0.5221089, -5.4649458, 5.6736479
2: -6.9344444, 4.9807320, -0.4780582, 0.7136130, -7.6480575, 5.4587903
3: -8.4231997, 4.0459538, -0.4969927, 0.4466695, -8.8698692, 4.5429463
4: -8.7769690, 5.0510368, -0.6103132, 0.6945386, -9.4715080, 5.6613503
5: -7.3778634, 4.3324661, -0.6252642, 0.6109307, -7.9887943, 4.9577303
6: -8.5039158, 4.3829861, -0.0333656, 1.3015839, -9.8055000, 4.4163518
7: -6.2551360, 6.3031368, -0.6434321, 0.6239196, -6.8790555, 6.9465690
8: -9.2558308, 4.3378139, -0.6138540, 0.6095831, -9.8654137, 4.9516678
9: -6.0188174, 6.0958581, -0.5549438, 0.6503330, -6.6691504, 6.6508017

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3654174, upper bound: 6.3017819
time: 1.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3530107, upper bound: 6.2615663
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3151188, 5.0462313, -1.1182098, 0.9202838, -8.2354031, 6.1644411
1: -5.1856198, 5.4158859, -0.9345390, 0.8959125, -6.0815325, 6.3504248
2: -7.2729502, 5.2057147, -0.9988400, 1.1763062, -8.4492569, 6.2045546
3: -8.8203754, 4.2304759, -1.1769482, 0.8068380, -9.6272135, 5.4074240
4: -9.1864567, 5.2778506, -1.3143544, 1.2029325, -10.3893890, 6.5922050
5: -7.7295513, 4.5377345, -1.1651046, 0.9937537, -8.7233047, 5.7028389
6: -8.9329529, 4.5965724, -0.9370954, 1.4597459, -10.3926983, 5.5336676
7: -6.5381107, 6.5868416, -1.1328228, 1.1643587, -7.7024693, 7.7196646
8: -9.7187939, 4.5761771, -1.3536148, 0.9758874, -10.6946812, 5.9297919
9: -6.2953672, 6.3807459, -1.0703744, 1.1594752, -7.4548426, 7.4511204

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2753427, upper bound: 6.3321306
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4357027, upper bound: 6.4375212
time: 1.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4345988, upper bound: 6.4005848
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.1658010, 3.5473330, -1.4376523, 1.0935880, -6.2593889, 4.9849854
1: -3.6093550, 3.8679023, -1.1143883, 1.1264780, -4.7358332, 4.9822907
2: -5.1136460, 3.7963769, -1.2991861, 1.3755157, -6.4891615, 5.0955629
3: -6.2369308, 3.0580466, -1.5913675, 0.9805315, -7.2174625, 4.6494141
4: -6.4826913, 3.8679109, -1.7068160, 1.3940835, -7.8767748, 5.5747271
5: -5.4552679, 3.2604082, -1.4893842, 1.1500008, -6.6052685, 4.7497921
6: -6.2125249, 3.3894362, -1.3595252, 1.5550243, -7.7675490, 4.7489614
7: -4.6971464, 4.7217884, -1.3971632, 1.4359207, -6.1330671, 6.1189518
8: -6.8303909, 3.2712245, -1.7568500, 1.1244302, -7.9548211, 5.0280743
9: -4.5199256, 4.5636482, -1.3356397, 1.4073119, -5.9272375, 5.8992882

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3626976, upper bound: 6.3492159
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3567521, upper bound: 6.3125853
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.4994006, 3.7930660, -2.4271965, 1.6849090, -7.1843095, 6.2202625
1: -3.8520403, 4.1162167, -1.7249579, 1.8759880, -5.7280283, 5.8411746
2: -5.4571419, 4.0187263, -2.3333192, 2.0164289, -7.4735708, 6.3520455
3: -6.6383553, 3.2404628, -2.8662047, 1.5323238, -8.1706791, 6.1066675
4: -6.9012947, 4.0979290, -2.9648361, 2.1020706, -9.0033655, 7.0627651
5: -5.8079453, 3.4673388, -2.5054226, 1.6917944, -7.4997396, 5.9727612
6: -6.6508002, 3.5952442, -2.7646813, 2.0163753, -8.6671753, 6.3599253
7: -4.9809837, 5.0101261, -2.3023312, 2.3076646, -7.2886486, 7.3124571
8: -7.2896943, 3.5155940, -3.1317172, 1.7027547, -8.9924488, 6.6473112
9: -4.8012891, 4.8493805, -2.2087915, 2.2355089, -7.0367980, 7.0581722

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3056622, upper bound: 6.4014245
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3056622, upper bound: 6.4742714
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.9803391, 4.8028984, -1.4376523, 1.0935880, -8.0739269, 6.2405510
1: -4.9428368, 5.1700234, -1.1143883, 1.1264780, -6.0693150, 6.2844119
2: -6.9344444, 4.9807320, -1.2991861, 1.3755157, -8.3099604, 6.2799182
3: -8.4231997, 4.0459538, -1.5913675, 0.9805315, -9.4037313, 5.6373215
4: -8.7769690, 5.0510368, -1.7068160, 1.3940835, -10.1710529, 6.7578526
5: -7.3778634, 4.3324661, -1.4893842, 1.1500008, -8.5278645, 5.8218503
6: -8.5039158, 4.3829861, -1.3595252, 1.5550243, -10.0589399, 5.7425113
7: -6.2551360, 6.3031368, -1.3971632, 1.4359207, -7.6910567, 7.7003002
8: -9.2558308, 4.3378139, -1.7568500, 1.1244302, -10.3802605, 6.0946636
9: -6.0188174, 6.0958581, -1.3356397, 1.4073119, -7.4261293, 7.4314976

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2338592, upper bound: 6.2554574
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4071627, upper bound: 6.3699548
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4014332, upper bound: 6.3245200
time: 2.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.64 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.7505200, upper bound: 6.3247220
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.7361955, upper bound: 6.2828303
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4572687
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4596387
IS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8056643, upper bound: 6.4557471
IS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8260071, upper bound: 6.4572687
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8000287, upper bound: 6.3258261
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8000287, upper bound: 6.4596387
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8371029, upper bound: 6.3921976
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8540009, upper bound: 6.3934877
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8534154, upper bound: 6.4965538
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8534154, upper bound: 6.4986144
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8240702, upper bound: 6.3919439
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8952090, upper bound: 6.4975010
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.8394065, upper bound: 6.3931464
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.9037576, upper bound: 6.4983196
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3588861, upper bound: 6.2609699
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3360330, upper bound: 6.2600941
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.4002567, upper bound: 6.4358400
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3993058, upper bound: 6.3993058
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3654174, upper bound: 6.3017819
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3530107, upper bound: 6.2615663
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.4357027, upper bound: 6.4375212
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.4345988, upper bound: 6.4005848
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3626976, upper bound: 6.3492159
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3567521, upper bound: 6.3125853
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3056622, upper bound: 6.4014245
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.3056622, upper bound: 6.4742714
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.4071627, upper bound: 6.3699548
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -6.4014332, upper bound: 6.3245200
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -6.4409483, upper bound: 6.4816400
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=7.834464073181152
rel_dist={6: [-7.107378872222906, 7.107378872222906]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7465193, upper bound: 6.5367070
time: 2.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5270972, upper bound: 6.5270972
time: 1.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.73
Output dim: 6, lower bound: -6.7465193, upper bound: 6.5367070
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.73
Output dim: 6, lower bound: -6.5270972, upper bound: 6.5270972

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.9998550, 2.8064208, -4.0230141, 2.8226054, -6.8224602, 6.8294349
1: -2.7389779, 3.0692990, -2.7556953, 3.0856514, -5.8246293, 5.8249941
2: -3.9650166, 3.0592949, -3.9878938, 3.0745039, -7.0395203, 7.0471888
3: -4.7972660, 2.4239097, -4.8247290, 2.4369347, -7.2342005, 7.2486386
4: -5.0186734, 3.1929450, -5.0468745, 3.2078700, -8.2265434, 8.2398195
5: -4.1977353, 2.6268024, -4.2217145, 2.6404290, -6.8381643, 6.8485169
6: -4.8564968, 2.9363008, -4.8847556, 2.9497087, -7.8062057, 7.8210564
7: -3.6791067, 3.6800561, -3.6985793, 3.6995137, -7.3786201, 7.3786354
8: -5.3407345, 2.7827613, -5.3712187, 2.7974010, -8.1381359, 8.1539803
9: -3.5578074, 3.5852487, -3.5766788, 3.6042769, -7.1620846, 7.1619272

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6236269, upper bound: 6.3844938
time: 1.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7045843, upper bound: 6.4913133
time: 2.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.9337635, 5.4761028, -3.9328716, 2.7594271, -10.6931906, 9.4089746
1: -5.6388774, 5.8590884, -2.6907589, 3.0218892, -8.6607666, 8.5498476
2: -7.8929882, 5.6129866, -3.8990443, 3.0151553, -10.9081440, 9.5120306
3: -9.5662737, 4.5686665, -4.7186203, 2.3860435, -11.9523172, 9.2872868
4: -9.9575558, 5.6808515, -4.9373150, 3.1497548, -13.1073103, 10.6181660
5: -8.3837214, 4.9044352, -4.1284008, 2.5874472, -10.9711685, 9.0328360
6: -9.7087727, 4.9445033, -4.7747536, 2.8974886, -12.6062613, 9.7192574
7: -7.0684695, 7.1211610, -3.6229706, 3.6239648, -10.6924343, 10.7441311
8: -10.5496216, 4.9511232, -5.2525463, 2.7399766, -13.2895985, 10.2036695
9: -6.8049579, 6.9037161, -3.5031438, 3.5306590, -10.3356171, 10.4068604

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4126207, upper bound: 6.3725300
time: 1.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4809345, upper bound: 6.4809345
time: 1.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.60 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -6.6236269, upper bound: 6.3844938
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -6.7045843, upper bound: 6.4913133
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -6.4126207, upper bound: 6.3725300
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 6, lower bound: -6.4809345, upper bound: 6.4809345

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.9844546, 2.0606411, -1.9739895, 1.3917928, -4.3762474, 4.0346308
1: -2.0740349, 2.2989137, -1.4383814, 1.5237949, -3.5978298, 3.7372952
2: -2.9153509, 2.3848472, -1.8479707, 1.7078478, -4.6231985, 4.2328176
3: -3.5709817, 1.8528252, -2.2851274, 1.2720140, -4.8429956, 4.1379528
4: -3.7332714, 2.4882493, -2.3665547, 1.7572510, -5.4905224, 4.8548040
5: -3.1084499, 2.0066948, -2.0402808, 1.4254801, -4.5339298, 4.0469756
6: -3.5162995, 2.3178573, -2.0989299, 1.7773043, -5.2936039, 4.4167871
7: -2.8111873, 2.8079703, -1.8793011, 1.9025557, -4.7137432, 4.6872711
8: -3.9201982, 2.0202591, -2.4642615, 1.4045532, -5.3247514, 4.4845209
9: -2.6934085, 2.7194502, -1.8036227, 1.8334591, -4.5268679, 4.5230732

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5772284, upper bound: 6.3816927
time: 1.89 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
time: 3.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.6458371, 2.5477169, -2.9927118, 2.0725789, -5.7184162, 5.5404286
1: -2.4847224, 2.8118055, -2.0768807, 2.3085706, -4.7932930, 4.8886862
2: -3.6115546, 2.8240449, -2.9298670, 2.3931098, -6.0046644, 5.7539120
3: -4.3844652, 2.2238281, -3.5823157, 1.8590199, -6.2434850, 5.8061438
4: -4.5874877, 2.9544845, -3.7347963, 2.4927833, -7.0802708, 6.6892805
5: -3.8301148, 2.4116855, -3.1216612, 2.0134020, -5.8435168, 5.5333467
6: -4.4072266, 2.7135110, -3.5275345, 2.3184190, -6.7256455, 6.2410455
7: -3.3834510, 3.3819151, -2.8147407, 2.8129900, -6.1964407, 6.1966558
8: -4.8654184, 2.5277309, -3.9446101, 2.0561862, -6.9216046, 6.4723411
9: -3.2666430, 3.2931833, -2.7039702, 2.7331760, -5.9998188, 5.9971533

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6386429, upper bound: 6.4857467
time: 3.19 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6386429, upper bound: 6.4913133
time: 2.86 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9747686, 4.7869244, -1.8881776, 1.3433211, -8.3180895, 6.6751022
1: -4.9404278, 5.1601496, -1.3862855, 1.4601066, -6.4005346, 6.5464354
2: -6.9273181, 4.9739866, -1.7593837, 1.6552029, -8.5825214, 6.7333703
3: -8.4223385, 4.0430822, -2.1752243, 1.2255681, -9.6479063, 6.2183065
4: -8.7726917, 5.0356045, -2.2622182, 1.6994042, -10.4720955, 7.2978230
5: -7.3784413, 4.3195138, -1.9527584, 1.3809650, -8.7594061, 6.2722721
6: -8.4844189, 4.3414345, -1.9834691, 1.7397999, -10.2242184, 6.3249035
7: -6.2542329, 6.3046980, -1.8016275, 1.8277081, -8.0819407, 8.1063251
8: -9.2427530, 4.2940741, -2.3506360, 1.3596246, -10.6023779, 6.6447101
9: -6.0149879, 6.0953374, -1.7282209, 1.7655051, -7.7804928, 7.8235583

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3693308, upper bound: 6.3587554
time: 1.44 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
time: 1.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.6094475, 5.2431564, -2.9146810, 2.0172253, -9.6266727, 8.1578369
1: -5.4025970, 5.6231365, -2.0281174, 2.2490625, -7.6516595, 7.6512537
2: -7.5668259, 5.3961139, -2.8472204, 2.3407307, -9.9075565, 8.2433338
3: -9.1813202, 4.3907728, -3.4832551, 1.8139502, -10.9952707, 7.8740282
4: -9.5567713, 5.4614425, -3.6287646, 2.4387226, -11.9954939, 9.0902071
5: -8.0436497, 4.7066174, -3.0338380, 1.9692122, -10.0128622, 7.7404556
6: -9.2933321, 4.7400913, -3.4233158, 2.2761447, -11.5694771, 8.1634073
7: -6.7931066, 6.8454227, -2.7438760, 2.7431772, -9.5362835, 9.5892982
8: -10.1061096, 4.7299194, -3.8325813, 2.0068936, -12.1130028, 8.5625010
9: -6.5381222, 6.6293135, -2.6352365, 2.6645236, -9.2026463, 9.2645502

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3725300, upper bound: 6.4126207
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3725300, upper bound: 6.4809345
time: 1.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.56 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.5772284, upper bound: 6.3816927
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.6386429, upper bound: 6.4857467
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.6386429, upper bound: 6.4913133
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.3693308, upper bound: 6.3587554
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.3725300, upper bound: 6.4126207
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.56
Output dim: 6, lower bound: -6.3725300, upper bound: 6.4809345

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2.7299247, 1.8816528, -1.4286773, 1.0894588, -3.8193836, 3.3103302
1: -1.9163861, 2.1043754, -1.1083400, 1.1214671, -3.0378532, 3.2127154
2: -2.6474628, 2.2142541, -1.2910005, 1.3713034, -4.0187664, 3.5052547
3: -3.2481058, 1.7050579, -1.5795662, 0.9771242, -4.2252302, 3.2846241
4: -3.3878231, 2.3142271, -1.6960649, 1.3887161, -4.7765393, 4.0102921
5: -2.8212483, 1.8629729, -1.4817519, 1.1454542, -3.9667025, 3.3447247
6: -3.1765306, 2.1834965, -1.3504941, 1.5506085, -4.7271390, 3.5339906
7: -2.5797119, 2.5787227, -1.3892689, 1.4297900, -4.0095019, 3.9679916
8: -3.5560427, 1.8656662, -1.7456365, 1.1202958, -4.6763382, 3.6113026
9: -2.4695835, 2.4964545, -1.3289682, 1.4016124, -3.8711958, 3.8254228

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5408971, upper bound: 6.3645133
time: 2.94 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
time: 1.86 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.7605360, 1.9030709, -3.6779327, 2.3790522, -5.1395884, 5.5810037
1: -1.9350173, 2.1278896, -2.4842789, 2.8091822, -4.7441998, 4.6121683
2: -2.6794872, 2.2346556, -3.6175148, 2.7784882, -5.4579754, 5.8521705
3: -3.2870529, 1.7230184, -4.4754286, 2.2084165, -5.4954691, 6.1984472
4: -3.4291561, 2.3341084, -4.4464836, 2.9511497, -6.3803058, 6.7805920
5: -2.8563850, 1.8798525, -3.7912683, 2.3276639, -5.1840487, 5.6711206
6: -3.2167311, 2.1985068, -4.4691095, 2.5404019, -5.7571330, 6.6676164
7: -2.6075382, 2.6066465, -3.4384446, 3.4196157, -6.0271540, 6.0450912
8: -3.5999737, 1.8815844, -4.7528410, 2.3063831, -5.9063568, 6.6344252
9: -2.4966273, 2.5232649, -3.3169780, 3.1982899, -5.6949172, 5.8402429

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
time: 2.81 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
time: 2.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.9551439, 1.3809321, -2.9927118, 2.0725789, -4.0277228, 4.3736439
1: -1.4268676, 1.5098075, -2.0768807, 2.3085706, -3.7354383, 3.5866880
2: -1.8285166, 1.6962208, -2.9298670, 2.3931098, -4.2216263, 4.6260877
3: -2.2609115, 1.2618241, -3.5823157, 1.8590199, -4.1199312, 4.8441401
4: -2.3436060, 1.7444773, -3.7347963, 2.4927833, -4.8363895, 5.4792738
5: -2.0210443, 1.4156493, -3.1216612, 2.0134020, -4.0344462, 4.5373106
6: -2.0735373, 1.7688398, -3.5275345, 2.3184190, -4.3919563, 5.2963743
7: -1.8622252, 1.8860421, -2.8147407, 2.8129900, -4.6752152, 4.7007828
8: -2.4392333, 1.3946896, -3.9446101, 2.0561862, -4.4954195, 5.3392997
9: -1.7870498, 1.8185271, -2.7039702, 2.7331760, -4.5202255, 4.5224972

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5724338, upper bound: 6.4476038
time: 4.49 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5532848, upper bound: 6.4470871
time: 3.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.9726520, 2.0583715, -2.9927118, 2.0725789, -5.0452309, 5.0510836
1: -2.0643368, 2.2932923, -2.0768807, 2.3085706, -4.3729076, 4.3701730
2: -2.9086456, 2.3796589, -2.9298670, 2.3931098, -5.3017554, 5.3095260
3: -3.5568466, 1.8474431, -3.5823157, 1.8590199, -5.4158664, 5.4297590
4: -3.7075248, 2.4788980, -3.7347963, 2.4927833, -6.2003078, 6.2136946
5: -3.0990927, 2.0020547, -3.1216612, 2.0134020, -5.1124945, 5.1237159
6: -3.5007610, 2.3075459, -3.5275345, 2.3184190, -5.8191800, 5.8350801
7: -2.7965198, 2.7950439, -2.8147407, 2.8129900, -5.6095095, 5.6097846
8: -3.9158545, 2.0435970, -3.9446101, 2.0561862, -5.9720407, 5.9882069
9: -2.6863189, 2.7155519, -2.7039702, 2.7331760, -5.4194946, 5.4195223

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6106215, upper bound: 6.4913133
time: 2.25 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6020577, upper bound: 6.4907674
time: 2.78 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -6.7003150, 4.5969229, -1.3594170, 1.0525129, -7.7528276, 5.9563398
1: -4.7381949, 4.9632907, -1.0684605, 1.0715057, -5.8097005, 6.0317512
2: -6.6507053, 4.7952428, -1.2245181, 1.3284812, -7.9791865, 6.0197611
3: -8.0873203, 3.8941078, -1.4903727, 0.9405119, -9.0278320, 5.3844805
4: -8.4258127, 4.8569822, -1.6114318, 1.3423084, -9.7681208, 6.4684143
5: -7.0881772, 4.1571717, -1.4126158, 1.1104387, -8.1986160, 5.5697875
6: -8.1397724, 4.1916499, -1.2549832, 1.5245130, -9.6642857, 5.4466333
7: -6.0173712, 6.0656114, -1.3280506, 1.3722186, -7.3895898, 7.3936620
8: -8.8766527, 4.1323214, -1.6576132, 1.0850381, -9.9616909, 5.7899346
9: -5.7873678, 5.8633776, -1.2695607, 1.3493257, -7.1366935, 7.1329384

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
time: 1.51 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3693308, upper bound: 6.3587554
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -6.7181273, 4.6086502, -3.6051965, 2.2951818, -9.0133095, 8.2138462
1: -4.7515659, 4.9756250, -2.4291692, 2.7497957, -7.5013618, 7.4047942
2: -6.6684675, 4.8068180, -3.5397074, 2.7245479, -9.3930149, 8.3465252
3: -8.1093140, 3.9039147, -4.3745790, 2.1653876, -10.2747021, 8.2784939
4: -8.4481926, 4.8683152, -4.3513384, 2.8981776, -11.3463707, 9.2196541
5: -7.1071677, 4.1672101, -3.6985860, 2.2860138, -9.3931818, 7.8657961
6: -8.1619396, 4.1999450, -4.3734570, 2.4845719, -10.6465111, 8.5734024
7: -6.0329752, 6.0813804, -3.3739326, 3.3410604, -9.3740358, 9.4553127
8: -8.9001541, 4.1410446, -4.6446705, 2.2685261, -11.1686802, 8.7857151
9: -5.8021154, 5.8786592, -3.2456470, 3.1356514, -8.9377670, 9.1243057

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
time: 1.39 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.8714604, 4.0250745, -2.9146810, 2.0172253, -7.8886857, 6.9397554
1: -4.1219864, 4.3728075, -2.0281174, 2.2490625, -6.3710489, 6.4009247
2: -5.8169184, 4.2467909, -2.8472204, 2.3407307, -8.1576490, 7.0940113
3: -7.1066451, 3.4408593, -3.4832551, 1.8139502, -8.9205952, 6.9241142
4: -7.3656635, 4.2814665, -3.6287646, 2.4387226, -9.8043861, 7.9102311
5: -6.2275739, 3.6615520, -3.0338380, 1.9692122, -8.1967859, 6.6953897
6: -7.0415173, 3.7057540, -3.4233158, 2.2761447, -9.3176622, 7.1290698
7: -5.3012037, 5.3446093, -2.7438760, 2.7431772, -8.0443811, 8.0884857
8: -7.7750497, 3.5895064, -3.8325813, 2.0068936, -9.7819433, 7.4220877
9: -5.1034951, 5.1642504, -2.6352365, 2.6645236, -7.7680187, 7.7994871

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3479178, upper bound: 6.3693308
time: 1.64 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3195022, upper bound: 6.3638576
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.9253788, 4.7532611, -2.9146810, 2.0172253, -8.9426041, 7.6679420
1: -4.9038134, 5.1236477, -2.0281174, 2.2490625, -7.1528759, 7.1517649
2: -6.8791294, 4.9415121, -2.8472204, 2.3407307, -9.2198601, 7.7887325
3: -8.3664522, 4.0163488, -3.4832551, 1.8139502, -10.1804028, 7.4996042
4: -8.7064972, 4.9995985, -3.6287646, 2.4387226, -11.1452198, 8.6283627
5: -7.3274937, 4.2894773, -3.0338380, 1.9692122, -9.2967062, 7.3233156
6: -8.4150848, 4.3096442, -3.4233158, 2.2761447, -10.6912298, 7.7329597
7: -6.2122359, 6.2608805, -2.7438760, 2.7431772, -8.9554129, 9.0047569
8: -9.1764841, 4.2694435, -3.8325813, 2.0068936, -11.1833782, 8.1020250
9: -5.9747615, 6.0541635, -2.6352365, 2.6645236, -8.6392851, 8.6893997

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3216371, upper bound: 6.4408270
time: 1.90 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3709997, upper bound: 6.4809220
time: 1.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 12.43 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.5408971, upper bound: 6.3645133
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.5724338, upper bound: 6.4476038
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.5532848, upper bound: 6.4470871
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.6106215, upper bound: 6.4913133
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.6020577, upper bound: 6.4907674
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3693308, upper bound: 6.3587554
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3479178, upper bound: 6.3693308
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3195022, upper bound: 6.3638576
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3216371, upper bound: 6.4408270
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.43
Output dim: 6, lower bound: -6.3709997, upper bound: 6.4809220

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -2.4835458, 1.7159245, -0.9088726, 0.8172740, -3.3008199, 2.6247971
1: -1.7612300, 1.9164410, -0.8035347, 0.7625140, -2.5237441, 2.7199757
2: -2.3867235, 2.0515122, -0.8157687, 1.0365465, -3.4232700, 2.8672810
3: -2.9334810, 1.5638515, -0.9055883, 0.7009611, -3.6344421, 2.4694397
4: -3.0504484, 2.1460986, -1.0536817, 1.0439225, -4.0943708, 3.1997805
5: -2.5586119, 1.7235268, -0.9723119, 0.8839162, -3.4425280, 2.6958387
6: -2.8473029, 2.0502796, -0.6309543, 1.3823044, -4.2296076, 2.6812339
7: -2.3551636, 2.3600006, -0.9558797, 1.0009944, -3.3561580, 3.3158803
8: -3.2024984, 1.7184117, -1.0867182, 0.8639265, -4.0664248, 2.8051300
9: -2.2540498, 2.2799909, -0.8939885, 1.0064757, -3.2605255, 3.1739795

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5408971, upper bound: 6.3645133
time: 2.91 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5408971, upper bound: 6.3645133
time: 2.78 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -2.4819090, 1.7147294, -2.1260533, 1.4650906, -3.9469995, 3.8407826
1: -1.7598938, 1.9150217, -1.5167010, 1.6191180, -3.3790116, 3.4317226
2: -2.3845456, 2.0503144, -1.9606266, 1.7968773, -4.1814227, 4.0109410
3: -2.9305532, 1.5630662, -2.4757228, 1.3498828, -4.2804360, 4.0387888
4: -3.0476623, 2.1449137, -2.5457630, 1.8501248, -4.8977871, 4.6906767
5: -2.5569730, 1.7226636, -2.1798725, 1.4966257, -4.0535984, 3.9025362
6: -2.8461225, 2.0487432, -2.2979281, 1.8180510, -4.6641736, 4.3466711
7: -2.3532183, 2.3588295, -2.0034935, 2.0080264, -4.3612447, 4.3623228
8: -3.1999454, 1.7179050, -2.6264782, 1.4770833, -4.6770287, 4.3443832
9: -2.2522960, 2.2784758, -1.9253119, 1.9339538, -4.1862497, 4.2037878

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
time: 1.99 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2.3831799, 1.6505792, -3.6779327, 2.3790522, -4.7622318, 5.3285122
1: -1.6992215, 1.8397032, -2.4842789, 2.8091822, -4.5084038, 4.3239822
2: -2.2805614, 1.9857086, -3.6175148, 2.7784882, -5.0590496, 5.6032233
3: -2.8051023, 1.5064721, -4.4754286, 2.2084165, -5.0135188, 5.9819007
4: -2.9118133, 2.0769000, -4.4464836, 2.9511497, -5.8629627, 6.5233836
5: -2.4558558, 1.6658444, -3.7912683, 2.3276639, -4.7835197, 5.4571128
6: -2.7126777, 1.9961900, -4.4691095, 2.5404019, -5.2530794, 6.4652996
7: -2.2632034, 2.2708125, -3.4384446, 3.4196157, -5.6828194, 5.7092571
8: -3.0577738, 1.6617162, -4.7528410, 2.3063831, -5.3641567, 6.4145575
9: -2.1661630, 2.1919351, -3.3169780, 3.1982899, -5.3644528, 5.5089130

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
time: 3.12 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5657397, upper bound: 6.3342837
time: 2.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.7846546, 3.3229728, -3.6779327, 2.3790522, -7.1637068, 7.0009055
1: -3.1924853, 3.6816967, -2.4842789, 2.8091822, -6.0016675, 6.1659756
2: -4.8088546, 3.6023817, -3.6175148, 2.7784882, -7.5873427, 7.2198963
3: -5.8291979, 2.8979268, -4.4754286, 2.2084165, -8.0376148, 7.3733554
4: -6.1902432, 3.7564554, -4.4464836, 2.9511497, -9.1413927, 8.2029390
5: -5.1185837, 3.0366340, -3.7912683, 2.3276639, -7.4462476, 6.8279023
6: -5.9611621, 3.3076329, -4.4691095, 2.5404019, -8.5015640, 7.7767425
7: -4.4480391, 4.4234538, -3.4384446, 3.4196157, -7.8676548, 7.8618984
8: -6.4985256, 3.1706748, -4.7528410, 2.3063831, -8.8049088, 7.9235158
9: -4.2743926, 4.3025904, -3.3169780, 3.1982899, -7.4726825, 7.6195683

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5532848, upper bound: 6.3335813
time: 3.52 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5532848, upper bound: 6.3344035
time: 2.73 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -1.4108059, 1.0798445, -2.7392614, 1.8942873, -3.3050933, 3.8191059
1: -1.0979445, 1.1085818, -1.9199570, 2.1148062, -3.2127507, 3.0285387
2: -1.2738465, 1.3602531, -2.6630859, 2.2232034, -3.4970498, 4.0233393
3: -1.5564719, 0.9676427, -3.2610078, 1.7118998, -3.2683716, 4.2286506
4: -1.6742289, 1.3766077, -3.3909409, 2.3187177, -3.9929466, 4.7675486
5: -1.4639034, 1.1362921, -2.8357644, 1.8702348, -3.3341384, 3.9720564
6: -1.3258181, 1.5438207, -3.1891162, 2.1844840, -3.5103021, 4.7329369
7: -1.3733447, 1.4149382, -2.5843959, 2.5848825, -3.9582272, 3.9993341
8: -1.7229097, 1.1110992, -3.5818474, 1.9015285, -3.6244383, 4.6929464
9: -1.3136181, 1.3879080, -2.4811492, 2.5110061, -3.8246241, 3.8690572

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6108296, upper bound: 6.4476038
time: 2.92 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6034101, upper bound: 6.4470689
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.6581616, 2.3684809, -2.7675834, 1.9142318, -5.5723934, 5.1360645
1: -2.4667158, 2.7955933, -1.9371191, 2.1366606, -4.6033764, 4.7327123
2: -3.5943832, 2.7671928, -2.6928105, 2.2421637, -5.8365469, 5.4600034
3: -4.4518623, 2.1971428, -3.2969480, 1.7285495, -6.1804118, 5.4940910
4: -4.4174986, 2.9387743, -3.4289715, 2.3375623, -6.7550611, 6.3677459
5: -3.7658143, 2.3181190, -2.8683460, 1.8858856, -5.6517000, 5.1864653
6: -4.4445066, 2.5256901, -3.2264822, 2.1983175, -6.6428242, 5.7521725
7: -3.4218538, 3.4002759, -2.6100013, 2.6106396, -6.0324936, 6.0102773
8: -4.7230148, 2.2967923, -3.6228235, 1.9169856, -6.6400003, 5.9196157
9: -3.3008616, 3.1790781, -2.5062437, 2.5359750, -5.8368368, 5.6853218

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5929982, upper bound: 6.4470871
time: 2.11 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5929982, upper bound: 6.4470871
time: 2.75 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -2.3090937, 1.6076500, -2.7398331, 1.8948905, -4.2039843, 4.3474832
1: -1.6474338, 1.7868768, -1.9185748, 2.1160049, -3.7634387, 3.7054515
2: -2.2059414, 1.9390945, -2.6634159, 2.2237487, -4.4296904, 4.6025105
3: -2.7095261, 1.4662611, -3.2607303, 1.7129223, -4.4224482, 4.7269917
4: -2.7996538, 2.0226104, -3.3909903, 2.3189843, -5.1186380, 5.4136009
5: -2.3840399, 1.6261947, -2.8369935, 1.8709688, -4.2550087, 4.4631882
6: -2.6122680, 1.9490091, -3.1903090, 2.1824646, -4.7947326, 5.1393180
7: -2.1913385, 2.2048392, -2.5844121, 2.5863097, -4.7776480, 4.7892513
8: -2.9622445, 1.6430535, -3.5836554, 1.9015529, -4.8637972, 5.2267089
9: -2.1045485, 2.1320677, -2.4819963, 2.5121055, -4.6166539, 4.6140642

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6571956, upper bound: 6.4885515
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
time: 2.23 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -3.6468980, 2.5360484, -2.7398710, 1.8947973, -5.5416956, 5.2759194
1: -2.4816341, 2.8043573, -1.9183178, 2.1158822, -4.5975161, 4.7226753
2: -3.6176832, 2.8280301, -2.6630404, 2.2236485, -5.8413315, 5.4910707
3: -4.4064455, 2.2394400, -3.2600381, 1.7131317, -6.1195774, 5.4994783
4: -4.6157575, 2.9372330, -3.3906088, 2.3190060, -6.9347634, 6.3278418
5: -3.8652823, 2.3810072, -2.8371408, 1.8710747, -5.7363567, 5.2181482
6: -4.3997984, 2.6571193, -3.1913531, 2.1818373, -6.5816355, 5.8484726
7: -3.4021754, 3.4039133, -2.5840843, 2.5867295, -5.9889050, 5.9879975
8: -4.8796778, 2.4649310, -3.5834842, 1.9017715, -6.7814493, 6.0484152
9: -3.2751994, 3.3090894, -2.4816971, 2.5120659, -5.7872653, 5.7907867

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6483579, upper bound: 6.4880508
time: 2.72 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
time: 1.89 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -6.4428868, 4.4175754, -0.8527707, 0.7896734, -7.2325602, 5.2703462
1: -4.5473638, 4.7797289, -0.7708682, 0.7316675, -5.2790313, 5.5505972
2: -6.3902917, 4.6273022, -0.7738215, 0.9992473, -7.3895388, 5.4011235
3: -7.7736664, 3.7546220, -0.8351673, 0.6757151, -8.4493818, 4.5897894
4: -8.1000624, 4.6882629, -0.9888179, 1.0080841, -9.1081467, 5.6770811
5: -6.8155117, 4.0057983, -0.9226052, 0.8558809, -7.6713924, 4.9284034
6: -7.8152819, 4.0485930, -0.5579814, 1.3684893, -9.1837711, 4.6065745
7: -5.7948408, 5.8421721, -0.9112517, 0.9584073, -6.7532482, 6.7534237
8: -8.5325174, 3.9775302, -1.0219762, 0.8392829, -9.3718004, 4.9995065
9: -5.5736227, 5.6457825, -0.8495573, 0.9684620, -6.5420847, 6.4953399

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
time: 1.62 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -6.4333987, 4.4113188, -2.0577099, 1.4205825, -7.8539810, 6.4690285
1: -4.5399919, 4.7732191, -1.4647722, 1.5702193, -6.1102114, 6.2379913
2: -6.3805437, 4.6212816, -1.8928508, 1.7543886, -8.1349325, 6.5141325
3: -7.7615333, 3.7498341, -2.3838682, 1.3001983, -9.0617313, 6.1337023
4: -8.0877457, 4.6822176, -2.4626441, 1.7952970, -9.8830423, 7.1448617
5: -6.8057761, 4.0006080, -2.1092317, 1.4500772, -8.2558537, 6.1098394
6: -7.8047266, 4.0429316, -2.1942997, 1.7901480, -9.5948744, 6.2372313
7: -5.7863703, 5.8345251, -1.9278656, 1.9489939, -7.7353640, 7.7623906
8: -8.5201511, 3.9728479, -2.5337825, 1.4310509, -9.9512024, 6.5066304
9: -5.5656757, 5.6379099, -1.8575798, 1.8712895, -7.4369650, 7.4954896

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3141258, upper bound: 6.2924153
time: 1.49 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3693308, upper bound: 6.3587554
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.2144794, 4.2599711, -3.6051965, 2.2951818, -8.5096607, 7.8651676
1: -4.3794141, 4.6167173, -2.4291692, 2.7497957, -7.1292095, 7.0458865
2: -6.1603589, 4.4781570, -3.5397074, 2.7245479, -8.8849068, 8.0178642
3: -7.4970202, 3.6298401, -4.3745790, 2.1653876, -9.6624079, 8.0044193
4: -7.8110542, 4.5400963, -4.3513384, 2.8981776, -10.7092323, 8.8914347
5: -6.5730667, 3.8716280, -3.6985860, 2.2860138, -8.8590803, 7.5702143
6: -7.5284796, 3.9280157, -4.3734570, 2.4845719, -10.0130520, 8.3014727
7: -5.5971975, 5.6421385, -3.3739326, 3.3410604, -8.9382582, 9.0160713
8: -8.2280426, 3.8457227, -4.6446705, 2.2685261, -10.4965687, 8.4903927
9: -5.3839517, 5.4524117, -3.2456470, 3.1356514, -8.5196028, 8.6980591

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3043405, upper bound: 6.2577727
time: 1.44 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
time: 5.41 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -8.7670956, 6.0486174, -3.6051965, 2.2951818, -11.0622768, 9.6538143
1: -6.2632847, 6.4635720, -2.4291692, 2.7497957, -9.0130806, 8.8927412
2: -8.7414913, 6.1491485, -3.5397074, 2.7245479, -11.4660397, 9.6888561
3: -10.5975704, 5.0132742, -4.3745790, 2.1653876, -12.7629585, 9.3878536
4: -11.0623245, 6.2307248, -4.3513384, 2.8981776, -13.9605026, 10.5820637
5: -9.2645397, 5.3984241, -3.6985860, 2.2860138, -11.5505533, 9.0970097
6: -10.7720699, 5.3849344, -4.3734570, 2.4845719, -13.2566414, 9.7583914
7: -7.8004322, 7.8636489, -3.3739326, 3.3410604, -11.1414928, 11.2375813
8: -11.6467438, 5.3881192, -4.6446705, 2.2685261, -13.9152699, 10.0327892
9: -7.5063224, 7.6070242, -3.2456470, 3.1356514, -10.6419735, 10.8526707

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3043405, upper bound: 6.2577727
time: 1.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
time: 1.54 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.1194453, 3.5179441, -2.6635685, 1.8409090, -6.9603543, 6.1815128
1: -3.5674558, 3.8379185, -1.8728367, 2.0568218, -5.6242776, 5.7107553
2: -5.0642700, 3.7564597, -2.5829990, 2.1723614, -7.2366314, 6.3394585
3: -6.1998672, 3.0314193, -3.1652040, 1.6679910, -7.8678584, 6.1966233
4: -6.4151702, 3.7921619, -3.2880983, 2.2661750, -8.6813450, 7.0802603
5: -5.4347000, 3.2215271, -2.7506528, 1.8271557, -7.2618556, 5.9721799
6: -6.0950437, 3.3150346, -3.0876667, 2.1434696, -8.2385130, 6.4027014
7: -4.6512585, 4.6896739, -2.5157118, 2.5170307, -7.1682892, 7.2053857
8: -6.7781363, 3.1585822, -3.4729452, 1.8534399, -8.6315765, 6.6315274
9: -4.4828033, 4.5282192, -2.4147968, 2.4443521, -6.9271555, 6.9430161

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3134753, upper bound: 6.3172933
time: 2.10 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3587554, upper bound: 6.3693308
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -7.6469398, 5.2185960, -2.6916046, 1.8607490, -9.5076885, 7.9102006
1: -5.4374418, 5.6408558, -1.8898990, 2.0785608, -7.5160027, 7.5307550
2: -7.5921459, 5.4135895, -2.6125665, 2.1911538, -9.7832994, 8.0261555
3: -9.2266731, 4.4073763, -3.2008839, 1.6845531, -10.9112263, 7.6082602
4: -9.6399288, 5.4708228, -3.3259232, 2.2849054, -11.9248343, 8.7967463
5: -8.0815563, 4.7084894, -2.7830400, 1.8427216, -9.9242783, 7.4915295
6: -9.3282585, 4.6723967, -3.1247897, 2.1572194, -11.4854774, 7.7971864
7: -6.8392820, 6.8883634, -2.5411086, 2.5426488, -9.3819313, 9.4294720
8: -10.1316690, 4.6352634, -3.5136323, 1.8688031, -12.0004721, 8.1488953
9: -6.5656910, 6.6661167, -2.4396205, 2.4691885, -9.0348797, 9.1057377

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2888833, upper bound: 6.3099868
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3218153, upper bound: 6.3638576
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.3735132, 4.3660383, -1.0774144, 0.8997583, -7.2732716, 5.4434528
1: -4.4972181, 4.7271366, -0.9104202, 0.8687024, -5.3659205, 5.6375570
2: -6.3206692, 4.5790339, -0.9628187, 1.1497855, -7.4704547, 5.5418525
3: -7.6968651, 3.7144094, -1.1238666, 0.7851762, -8.4820414, 4.8382759
4: -8.0093403, 4.6377592, -1.2642124, 1.1754501, -9.1847906, 5.9019718
5: -6.7404995, 3.9621849, -1.1255695, 0.9736211, -7.7141204, 5.0877542
6: -7.7138281, 4.0063090, -0.8801841, 1.4476103, -9.1614380, 4.8864932
7: -5.7362709, 5.7771807, -1.1002698, 1.1322727, -6.8685436, 6.8774505
8: -8.4362392, 3.9328539, -1.3035662, 0.9538763, -9.3901157, 5.2364202
9: -5.5151577, 5.5857577, -1.0363007, 1.1287642, -6.6439219, 6.6220584

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4208864, upper bound: 6.4254336
time: 1.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4202310, upper bound: 6.3993154
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.6825805, 4.5836735, -2.3796015, 1.6534014, -8.3359814, 6.9632750
1: -4.7253885, 4.9489207, -1.6951379, 1.8395370, -6.5649257, 6.6440587
2: -6.6341329, 4.7827368, -2.2824540, 1.9849582, -8.6190910, 7.0651908
3: -8.0710316, 3.8841445, -2.8053143, 1.5051336, -9.5761652, 6.6894588
4: -8.3993673, 4.8408952, -2.8994236, 2.0690687, -10.4684362, 7.7403188
5: -7.0701160, 4.1453543, -2.4563017, 1.6645491, -8.7346649, 6.6016560
6: -8.1080046, 4.1756873, -2.7005732, 1.9902380, -10.0982428, 6.8762608
7: -6.0030794, 6.0490556, -2.2590036, 2.2658114, -8.2688904, 8.3080597
8: -8.8516521, 4.1226201, -3.0623271, 1.6733634, -10.5250158, 7.1849470
9: -5.7732062, 5.8487382, -2.1669846, 2.1932299, -7.9664364, 8.0157223

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4408305, upper bound: 6.4600315
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4408305, upper bound: 6.4809220
time: 1.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.23 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5408971, upper bound: 6.3645133
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5408971, upper bound: 6.3645133
IS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5673195, upper bound: 6.3344035
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5657397, upper bound: 6.3342837
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5532848, upper bound: 6.3335813
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5532848, upper bound: 6.3344035
IS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.6108296, upper bound: 6.4476038
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.6034101, upper bound: 6.4470689
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5929982, upper bound: 6.4470871
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.5929982, upper bound: 6.4470871
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.6571956, upper bound: 6.4885515
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.6483579, upper bound: 6.4880508
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
IS_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3141258, upper bound: 6.2924153
IS_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3693308, upper bound: 6.3587554
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3043405, upper bound: 6.2577727
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3043405, upper bound: 6.2577727
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3134753, upper bound: 6.3172933
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3587554, upper bound: 6.3693308
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.2888833, upper bound: 6.3099868
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.3218153, upper bound: 6.3638576
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.4208864, upper bound: 6.4254336
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.4202310, upper bound: 6.3993154
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.4408305, upper bound: 6.4600315
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 6, lower bound: -6.4408305, upper bound: 6.4809220

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.1547177, 1.5041445, -0.9088726, 0.8172740, -2.9719918, 2.4130173
1: -1.5554699, 1.6669428, -0.8035347, 0.7625140, -2.3179839, 2.4704776
2: -2.0396292, 1.8361528, -0.8157687, 1.0365465, -3.0761757, 2.6519215
3: -2.5123439, 1.3776386, -0.9055883, 0.7009611, -3.2133050, 2.2832270
4: -2.5965900, 1.9225529, -1.0536817, 1.0439225, -3.6405125, 2.9762347
5: -2.2223194, 1.5364794, -0.9723119, 0.8839162, -3.1062355, 2.5087912
6: -2.4061728, 1.8746271, -0.6309543, 1.3823044, -3.7884772, 2.5055814
7: -2.0538421, 2.0708575, -0.9558797, 1.0009944, -3.0548365, 3.0267372
8: -2.7330134, 1.5271206, -1.0867182, 0.8639265, -3.5969400, 2.6138387
9: -1.9663146, 1.9909869, -0.8939885, 1.0064757, -2.9727902, 2.8849754

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3639310
time: 3.54 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3645133
time: 2.50 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.4752817, 3.0276322, -0.9088726, 0.8172740, -5.2925558, 3.9365048
1: -3.0153890, 3.4515133, -0.8035347, 0.7625140, -3.7779031, 4.2550478
2: -4.5090575, 3.3726444, -0.8157687, 1.0365465, -5.5456038, 4.1884131
3: -5.4769955, 2.7092280, -0.9055883, 0.7009611, -6.1779566, 3.6148164
4: -5.8241148, 3.5529716, -1.0536817, 1.0439225, -6.8680372, 4.6066532
5: -4.6054773, 2.8702517, -0.9723119, 0.8839162, -5.4893937, 3.8425636
6: -5.5594406, 3.1611476, -0.6309543, 1.3823044, -6.9417448, 3.7921019
7: -4.1831141, 4.1275721, -0.9558797, 1.0009944, -5.1841087, 5.0834517
8: -6.0906067, 2.9129236, -1.0867182, 0.8639265, -6.9545331, 3.9996419
9: -4.0021195, 4.0453935, -0.8939885, 1.0064757, -5.0085955, 4.9393821

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3639310
time: 2.22 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3645133
time: 3.45 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2.1592999, 1.5069427, -2.1260533, 1.4650906, -3.6243906, 3.6329961
1: -1.5579374, 1.6703067, -1.5167010, 1.6191180, -3.1770554, 3.1870077
2: -2.0440252, 1.8390193, -1.9606266, 1.7968773, -3.8409023, 3.7996459
3: -2.5173473, 1.3804613, -2.4757228, 1.3498828, -3.8672302, 3.8561840
4: -2.6023524, 1.9255345, -2.5457630, 1.8501248, -4.4524775, 4.4712973
5: -2.2271390, 1.5391095, -2.1798725, 1.4966257, -3.7237647, 3.7189820
6: -2.4133246, 1.8761002, -2.2979281, 1.8180510, -4.2313757, 4.1740284
7: -2.0575657, 2.0752676, -2.0034935, 2.0080264, -4.0655918, 4.0787611
8: -2.7393923, 1.5300404, -2.6264782, 1.4770833, -4.2164755, 4.1565185
9: -1.9700632, 1.9949286, -1.9253119, 1.9339538, -3.9040170, 3.9202404

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5677006, upper bound: 6.3808963
time: 2.54 seconds

## Relational analysis of IS_A1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
time: 2.26 seconds

## BFS IS instance: IS_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4680781, 3.0228024, -2.1260533, 1.4650906, -5.9331689, 5.1488557
1: -3.0106053, 3.4458411, -1.5167010, 1.6191180, -4.6297235, 4.9625421
2: -4.5010524, 3.3677990, -1.9606266, 1.7968773, -6.2979298, 5.3284254
3: -5.4668784, 2.7052531, -2.4757228, 1.3498828, -6.8167610, 5.1809759
4: -5.8136315, 3.5481923, -2.5457630, 1.8501248, -7.6637564, 6.0939550
5: -4.5980349, 2.8663099, -2.1798725, 1.4966257, -6.0946608, 5.0461826
6: -5.5510416, 3.1567357, -2.2979281, 1.8180510, -7.3690925, 5.4546638
7: -4.1761651, 4.1215181, -2.0034935, 2.0080264, -6.1841917, 6.1250114
8: -6.0800190, 2.9094265, -2.6264782, 1.4770833, -7.5571022, 5.5359049
9: -3.9955056, 4.0389442, -1.9253119, 1.9339538, -5.9294596, 5.9642563

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5170995, upper bound: 6.3129999
time: 2.43 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
time: 3.01 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -1.7794100, 1.2835774, -3.4516120, 2.2482481, -4.0276580, 4.7351894
1: -1.3231959, 1.3877167, -2.3445683, 2.6412075, -3.9644034, 3.7322850
2: -1.6482201, 1.5997987, -3.3828287, 2.6388514, -4.2870712, 4.9826274
3: -2.0319405, 1.1730409, -4.1838408, 2.0860784, -4.1180191, 5.3568816
4: -2.1257176, 1.6712241, -4.1693168, 2.7988055, -4.9245234, 5.8405409
5: -1.8397199, 1.3367144, -3.5592420, 2.2101350, -4.0498548, 4.8959565
6: -1.8988305, 1.6978054, -4.1654119, 2.4367642, -4.3355947, 5.8632174
7: -1.7131139, 1.7438658, -3.2332330, 3.2215681, -4.9346819, 4.9770989
8: -2.2207544, 1.3129653, -4.4519196, 2.1873922, -4.4081469, 5.7648849
9: -1.6383169, 1.6830016, -3.1177001, 3.0181017, -4.6564188, 4.8007016

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5814750, upper bound: 6.3343629
time: 3.48 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5858405, upper bound: 6.3349824
time: 2.62 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -3.0985961, 2.1189392, -3.4521430, 2.2484760, -5.3470721, 5.5710821
1: -2.1435711, 2.3809063, -2.3445799, 2.6414924, -4.7850637, 4.7254863
2: -3.0365045, 2.4508905, -3.3830767, 2.6390319, -5.6755362, 5.8339672
3: -3.7158360, 1.9154236, -4.1836505, 2.0866218, -5.8024578, 6.0990744
4: -3.8894813, 2.5599241, -4.1693130, 2.7991686, -6.6886501, 6.7292371
5: -3.1986060, 2.0693650, -3.5598903, 2.2104077, -5.4090137, 5.6292553
6: -3.6665773, 2.3702621, -4.1669898, 2.4362397, -6.1028171, 6.5372519
7: -2.9092577, 2.9055712, -3.2333016, 3.2224550, -6.1317129, 6.1388731
8: -4.0887170, 2.0764346, -4.4526825, 2.1874719, -6.2761889, 6.5291171
9: -2.7882023, 2.8214009, -3.1179206, 3.0185077, -5.8067102, 5.9393215

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5683269, upper bound: 6.3340328
time: 2.60 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5683269, upper bound: 6.3347749
time: 2.07 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -3.7182097, 2.5670981, -3.6779327, 2.3790522, -6.0972619, 6.2450309
1: -2.5233345, 2.8632562, -2.4842789, 2.8091822, -5.3325167, 5.3475351
2: -3.6787031, 2.8762131, -3.6175148, 2.7784882, -6.4571915, 6.4937277
3: -4.4995480, 2.2797768, -4.4754286, 2.2084165, -6.7079644, 6.7552052
4: -4.7404003, 2.9885497, -4.4464836, 2.9511497, -7.6915503, 7.4350333
5: -3.9367988, 2.4160070, -3.7912683, 2.3276639, -6.2644625, 6.2072754
6: -4.4895716, 2.7010558, -4.4691095, 2.5404019, -7.0299735, 7.1701651
7: -3.4821761, 3.4731617, -3.4384446, 3.4196157, -6.9017916, 6.9116063
8: -4.9627061, 2.4081926, -4.7528410, 2.3063831, -7.2690892, 7.1610336
9: -3.3405929, 3.3580961, -3.3169780, 3.1982899, -6.5388827, 6.6750741

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5532848, upper bound: 6.3335813
time: 2.03 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5511437, upper bound: 6.3334789
time: 1.96 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.7485886, 3.3031445, -3.6779327, 2.3790522, -7.1276407, 6.9810772
1: -3.1678987, 3.6572843, -2.4842789, 2.8091822, -5.9770808, 6.1415634
2: -4.7761030, 3.5806215, -3.6175148, 2.7784882, -7.5545912, 7.1981363
3: -5.7851467, 2.8783305, -4.4754286, 2.2084165, -7.9935632, 7.3537588
4: -6.1319094, 3.7293258, -4.4464836, 2.9511497, -9.0830593, 8.1758099
5: -5.0823951, 3.0175333, -3.7912683, 2.3276639, -7.4100590, 6.8088017
6: -5.9126863, 3.2832847, -4.4691095, 2.5404019, -8.4530888, 7.7523942
7: -4.4117060, 4.3894434, -3.4384446, 3.4196157, -7.8313217, 7.8278880
8: -6.4587727, 3.1726971, -4.7528410, 2.3063831, -8.7651558, 7.9255381
9: -4.2458987, 4.2767758, -3.3169780, 3.1982899, -7.4441886, 7.5937538

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5046347, upper bound: 6.3112203
time: 4.18 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5511437, upper bound: 6.3342837
time: 2.50 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.8943728, 0.8101976, -2.4907384, 1.7270652, -2.6214380, 3.3009360
1: -0.7951559, 0.7542862, -1.7635958, 1.9250767, -2.7202327, 2.5178819
2: -0.8048317, 1.0268923, -2.3998940, 2.0589170, -2.8637488, 3.4267864
3: -0.8871552, 0.6943705, -2.9438982, 1.5694915, -2.4566467, 3.6382687
4: -1.0364995, 1.0347024, -3.0510640, 2.1482050, -3.1847045, 4.0857663
5: -0.9593268, 0.8766985, -2.5711193, 1.7294242, -2.6887510, 3.4478178
6: -0.6119295, 1.3787009, -2.8566880, 2.0501125, -2.6620421, 4.2353888
7: -0.9440537, 0.9899694, -2.3581583, 2.3644581, -3.3085117, 3.3481278
8: -1.0697770, 0.8574728, -3.2246687, 1.7515985, -2.8213754, 4.0821414
9: -0.8822150, 0.9967003, -2.2635570, 2.2924273, -3.1746423, 3.2602572

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6108296, upper bound: 6.4476038
time: 1.85 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6108296, upper bound: 6.4476038
time: 2.89 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -2.1089973, 1.4552153, -2.4945843, 1.7294459, -3.8384433, 3.9497995
1: -1.5041616, 1.6068910, -1.7656654, 1.9278558, -3.4320173, 3.3725564
2: -1.9417577, 1.7863898, -2.4035332, 2.0612864, -4.0030441, 4.1899233
3: -2.4537992, 1.3355407, -2.9480417, 1.5718799, -4.0256791, 4.2835822
4: -2.5250351, 1.8370106, -3.0558310, 2.1506815, -4.6757164, 4.8928413
5: -2.1629412, 1.4874437, -2.5751872, 1.7316577, -3.8945990, 4.0626307
6: -2.2724059, 1.8113627, -2.8627763, 2.0513692, -4.3237753, 4.6741390
7: -1.9883797, 1.9911011, -2.3612251, 2.3682323, -4.3566122, 4.3523264
8: -2.6030228, 1.4683601, -3.2299962, 1.7540224, -4.3570452, 4.6983562
9: -1.9086967, 1.9209532, -2.2666216, 2.2957213, -4.2044182, 4.1875749

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6034101, upper bound: 6.4470689
time: 2.44 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6034101, upper bound: 6.4470689
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -3.6581616, 2.3684809, -2.3945773, 1.6637946, -5.3219562, 4.7630582
1: -2.4667158, 2.7955933, -1.7043877, 1.8511310, -4.3178468, 4.4999809
2: -3.5943832, 2.7671928, -2.2977462, 1.9954298, -5.5898132, 5.0649390
3: -4.4518623, 2.1971428, -2.8215799, 1.5143440, -5.9662066, 5.0187225
4: -4.4174986, 2.9387743, -2.9190650, 2.0812912, -6.4987898, 5.8578396
5: -3.7658143, 2.3181190, -2.4724076, 1.6737338, -5.4395480, 4.7905264
6: -4.4445066, 2.5256901, -2.7266145, 1.9979925, -6.4424992, 5.2523046
7: -3.4218538, 3.4002759, -2.2706375, 2.2793338, -5.7011876, 5.6709137
8: -4.7230148, 2.2967923, -3.0844493, 1.6933284, -6.4163432, 5.3812418
9: -3.3008616, 3.1790781, -2.1791110, 2.2072754, -5.5081367, 5.3581891

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5733896, upper bound: 6.4466974
time: 2.80 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5882750, upper bound: 6.4466974
time: 2.50 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -3.6581616, 2.3684809, -4.7679958, 3.3167229, -6.9748845, 7.1364765
1: -2.4667158, 2.7955933, -3.1799309, 3.6720564, -6.1387720, 5.9755239
2: -3.5943832, 2.7671928, -4.7964897, 3.5935953, -7.1879787, 7.5636826
3: -4.4518623, 2.1971428, -5.8096571, 2.8895078, -7.3413701, 8.0067997
4: -4.4174986, 2.9387743, -6.1581564, 3.7426677, -8.1601658, 9.0969305
5: -3.7658143, 2.3181190, -5.1041279, 3.0284915, -6.7943058, 7.4222469
6: -4.4445066, 2.5256901, -5.9384780, 3.2937841, -7.7382908, 8.4641685
7: -3.4218538, 3.4002759, -4.4292674, 4.4067492, -7.8286028, 7.8295431
8: -4.7230148, 2.2967923, -6.4865208, 3.1846981, -7.9077129, 8.7833128
9: -3.3008616, 3.1790781, -4.2628822, 4.2937775, -7.5946388, 7.4419603

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5929982, upper bound: 6.4470871
time: 2.94 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5882750, upper bound: 6.4466974
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -2.0836437, 1.4635767, -2.1651609, 1.5148488, -3.5984926, 3.6287374
1: -1.5078440, 1.6150775, -1.5600426, 1.6765333, -3.1843772, 3.1751201
2: -1.9676986, 1.7911062, -2.0541716, 1.8443913, -3.8120899, 3.8452778
3: -2.4202583, 1.3383131, -2.5262775, 1.3839612, -3.8042195, 3.8645906
4: -2.4940903, 1.8693862, -2.6024358, 1.9237005, -4.4177909, 4.4718218
5: -2.1520047, 1.4982643, -2.2364733, 1.5429621, -3.6949668, 3.7347376
6: -2.3066711, 1.8358476, -2.4168515, 1.8766203, -4.1832914, 4.2526989
7: -1.9846188, 2.0054264, -2.0599394, 2.0769544, -4.0615730, 4.0653658
8: -2.6391554, 1.5112797, -2.7548349, 1.5577344, -4.1968899, 4.2661147
9: -1.9055743, 1.9334995, -1.9774296, 2.0043712, -3.9099455, 3.9109292

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
time: 2.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
time: 2.24 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -2.0922189, 1.4687072, -4.5069690, 3.1413944, -5.2336130, 5.9756761
1: -1.5130326, 1.6214635, -3.0232658, 3.4809103, -4.9939427, 4.6447296
2: -1.9766406, 1.7964687, -4.5330272, 3.4209716, -5.3976121, 6.3294959
3: -2.4311457, 1.3430344, -5.4857273, 2.7447283, -5.1758738, 6.8287616
4: -2.5045950, 1.8745084, -5.8180676, 3.5702381, -6.0748329, 7.6925759
5: -2.1609302, 1.5025568, -4.8204355, 2.8872569, -5.0481873, 6.3229923
6: -2.3175564, 1.8397013, -5.6030550, 3.1588011, -5.4763575, 7.4427562
7: -1.9922212, 2.0128558, -4.1959338, 4.1821785, -6.1743999, 6.2087898
8: -2.6514831, 1.5147905, -6.1235423, 3.0324857, -5.6839685, 7.6383328
9: -1.9131308, 1.9406891, -4.0326924, 4.0744867, -5.9876175, 5.9733815

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6531277, upper bound: 6.4523644
time: 2.90 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -3.4246662, 2.3788795, -2.1708910, 1.5183512, -4.9430175, 4.5497704
1: -2.3431275, 2.6345754, -1.5632341, 1.6807263, -4.0238538, 4.1978092
2: -3.3828683, 2.6789479, -2.0597601, 1.8479526, -5.2308207, 4.7387080
3: -4.1236906, 2.1106935, -2.5327740, 1.3874285, -5.5111189, 4.6434674
4: -4.3135548, 2.7838430, -2.6097836, 1.9274098, -6.2409644, 5.3936267
5: -3.6145804, 2.2551677, -2.2424531, 1.5462302, -5.1608105, 4.4976206
6: -4.1030426, 2.5381565, -2.4254544, 1.8786352, -5.9816780, 4.9636106
7: -3.1997325, 3.2039413, -2.0647295, 2.0823960, -5.2821283, 5.2686710
8: -4.5615754, 2.3284960, -2.7627468, 1.5611458, -6.1227212, 5.0912428
9: -3.0793426, 3.1135719, -1.9821534, 2.0092826, -5.0886250, 5.0957251

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
time: 2.55 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
time: 2.03 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -3.4186418, 2.3745775, -4.5065508, 3.1410079, -6.5596495, 6.8811283
1: -2.3393724, 2.6299415, -3.0226998, 3.4804540, -5.8198261, 5.6526413
2: -3.3764939, 2.6748490, -4.5321984, 3.4205663, -6.7970600, 7.2070475
3: -4.1161356, 2.1071308, -5.4844027, 2.7446761, -6.8608117, 7.5915337
4: -4.3052025, 2.7795811, -5.8170204, 3.5700374, -7.8752398, 8.5966015
5: -3.6078379, 2.2515678, -4.8200502, 2.8871443, -6.4949822, 7.0716181
6: -4.0948629, 2.5348363, -5.6036406, 3.1578977, -7.2527609, 8.1384773
7: -3.1942587, 3.1984212, -4.1952162, 4.1822062, -7.3764648, 7.3936377
8: -4.5531516, 2.3236895, -6.1227608, 3.0326614, -7.5858130, 8.4464502
9: -3.0740976, 3.1082511, -4.0319953, 4.0740585, -7.1481562, 7.1402464

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6443998, upper bound: 6.4516106
time: 2.38 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9585714, 4.0813875, -0.8527707, 0.7896734, -6.7482448, 4.9341583
1: -4.1894178, 4.4349203, -0.7708682, 0.7316675, -4.9210854, 5.2057886
2: -5.9011717, 4.3109431, -0.7738215, 0.9992473, -6.9004188, 5.0847645
3: -7.1864724, 3.4909163, -0.8351673, 0.6757151, -7.8621874, 4.3260837
4: -7.4869843, 4.3720770, -0.9888179, 1.0080841, -8.4950686, 5.3608952
5: -6.3015003, 3.7218285, -0.9226052, 0.8558809, -7.1573811, 4.6444335
6: -7.2053380, 3.7863364, -0.5579814, 1.3684893, -8.5738277, 4.3443179
7: -5.3755569, 5.4199219, -0.9112517, 0.9584073, -6.3339643, 6.3311734
8: -7.8856792, 3.6916826, -1.0219762, 0.8392829, -8.7249622, 4.7136588
9: -5.1712446, 5.2360034, -0.8495573, 0.9684620, -6.1397066, 6.0855608

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.2901621, upper bound: 6.2925732
time: 1.51 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.4962101, 5.8691406, -0.8527707, 0.7896734, -9.2858839, 6.7219114
1: -6.0722098, 6.2500081, -0.7708682, 0.7316675, -6.8038774, 7.0208764
2: -8.4766512, 5.9809580, -0.7738215, 0.9992473, -9.4758987, 6.7547793
3: -10.2573805, 4.8737402, -0.8351673, 0.6757151, -10.9330959, 5.7089076
4: -10.7272243, 6.0622001, -0.9888179, 1.0080841, -11.7353086, 7.0510178
5: -8.9918995, 5.2223492, -0.9226052, 0.8558809, -9.8477802, 6.1449542
6: -10.4479151, 5.2209554, -0.5579814, 1.3684893, -11.8164043, 5.7789369
7: -7.5778170, 7.6362500, -0.9112517, 0.9584073, -8.5362244, 8.5475016
8: -11.2922010, 5.2327867, -1.0219762, 0.8392829, -12.1314840, 6.2547626
9: -7.2881656, 7.3876328, -0.8495573, 0.9684620, -8.2566280, 8.2371902

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3109902, upper bound: 6.3314513
time: 1.50 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3109902, upper bound: 6.3429029
time: 2.29 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -5.8887143, 4.0313282, -0.5851049, 0.6231811, -6.5118952, 4.6164331
1: -4.1376052, 4.3830957, -0.5722691, 0.5721124, -4.7097178, 4.9553647
2: -5.8292208, 4.2630749, -0.5396478, 0.7802668, -6.6074257, 4.8027229
3: -7.1042848, 3.4509928, -0.5659034, 0.5042460, -7.6085310, 4.0168962
4: -7.3997641, 4.3245306, -0.7035764, 0.7543510, -8.1541147, 5.0281072
5: -6.2265649, 3.6784706, -0.6993151, 0.6656973, -6.8922620, 4.3777857
6: -7.1110210, 3.7468462, -0.1332172, 1.3186508, -8.4296722, 3.8800635
7: -5.3156037, 5.3568568, -0.6991035, 0.7080489, -6.0236526, 6.0559602
8: -7.7901287, 3.6398377, -0.7047739, 0.6569430, -8.4470720, 4.3446116
9: -5.1122532, 5.1754446, -0.6218306, 0.7362934, -5.8485465, 5.7972751

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2200490, upper bound: 6.2360736
time: 1.69 seconds

## Relational analysis of IS_A2_B1_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -6.2087325, upper bound: 6.1945137
time: 1.66 seconds

## BFS IS instance: IS_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -6.1878371, 4.2389727, -1.5597955, 1.1585765, -7.3464136, 5.7987680
1: -4.3588018, 4.5977693, -1.1849846, 1.2110348, -5.5698366, 5.7827539
2: -6.1317635, 4.4599924, -1.4170933, 1.4473685, -7.5791321, 5.8770857
3: -7.4640822, 3.6155777, -1.7469521, 1.0417371, -8.5058193, 5.3625298
4: -7.7764244, 4.5210743, -1.8574916, 1.4655460, -9.2419701, 6.3785658
5: -6.5442319, 3.8559091, -1.6144334, 1.2043446, -7.7485766, 5.4703426
6: -7.4929514, 3.9084129, -1.5134120, 1.6016185, -9.0945702, 5.4218249
7: -5.5739856, 5.6197958, -1.4975127, 1.5342166, -7.1082020, 7.1173086
8: -8.1908140, 3.8236749, -1.9069271, 1.1832716, -9.3740854, 5.7306023
9: -5.3611965, 5.4295096, -1.4363478, 1.4983499, -6.8595467, 6.8658576

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3172933, upper bound: 6.3134753
time: 1.61 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.3172933, upper bound: 6.3587554
time: 1.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.80 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3639310
IS_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3645133
IS_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3639310
IS_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5269688, upper bound: 6.3645133
IS_A1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5677006, upper bound: 6.3808963
IS_A1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
IS_A1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5170995, upper bound: 6.3129999
IS_A1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5738627, upper bound: 6.3815380
IS_A1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5814750, upper bound: 6.3343629
IS_A1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5858405, upper bound: 6.3349824
IS_A1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5683269, upper bound: 6.3340328
IS_A1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5683269, upper bound: 6.3347749
IS_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5532848, upper bound: 6.3335813
IS_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5511437, upper bound: 6.3334789
IS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5046347, upper bound: 6.3112203
IS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5511437, upper bound: 6.3342837
IS_A1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6108296, upper bound: 6.4476038
IS_A1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6108296, upper bound: 6.4476038
IS_A1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6034101, upper bound: 6.4470689
IS_A1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6034101, upper bound: 6.4470689
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5733896, upper bound: 6.4466974
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5882750, upper bound: 6.4466974
IS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5929982, upper bound: 6.4470871
IS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.5882750, upper bound: 6.4466974
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
IS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
IS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6531277, upper bound: 6.4523644
IS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6545131, upper bound: 6.4526872
IS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
IS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
IS_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6443998, upper bound: 6.4516106
IS_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.6460873, upper bound: 6.4519910
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.2901621, upper bound: 6.2925732
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.3550728, upper bound: 6.3429029
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.3109902, upper bound: 6.3314513
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.3109902, upper bound: 6.3429029
IS_A2_B1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.2200490, upper bound: 6.2360736
IS_A2_B1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.2087325, upper bound: 6.1945137
IS_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.3172933, upper bound: 6.3134753
IS_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 6, lower bound: -6.3172933, upper bound: 6.3587554
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3043405, upper bound: 6.2577727
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3043405, upper bound: 6.2577727
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3638576, upper bound: 6.3218153
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3134753, upper bound: 6.3172933
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3587554, upper bound: 6.3693308
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.2888833, upper bound: 6.3099868
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.3218153, upper bound: 6.3638576
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.4208864, upper bound: 6.4254336
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.4202310, upper bound: 6.3993154
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.4408305, upper bound: 6.4600315
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 6, lower bound: -6.4408305, upper bound: 6.4809220
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=7.834464073181152
rel_dist={6: [-7.107152351905672, 7.10715235190567]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1054814, upper bound: 7.1053660
time: 5.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.1053195, upper bound: 7.1053195
time: 3.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.12
Output dim: 6, lower bound: -7.1054814, upper bound: 7.1053660
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.12
Output dim: 6, lower bound: -7.1053195, upper bound: 7.1053195

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.2819676, 2.3088839, -3.4686823, 2.4388800, -5.7208476, 5.7775660
1: -2.2529716, 2.5541646, -2.3659348, 2.6905828, -4.9435544, 4.9200993
2: -3.2449141, 2.5965114, -3.4349506, 2.7155809, -5.9604950, 6.0314617
3: -3.9315817, 2.0288391, -4.1601114, 2.1325989, -6.0641804, 6.1889505
4: -4.1165495, 2.7239854, -4.3556237, 2.8475165, -6.9640660, 7.0796089
5: -3.4357038, 2.2112360, -3.6368489, 2.3182571, -5.7539606, 5.8480849
6: -3.9629970, 2.5396757, -4.1991653, 2.6405892, -6.6035862, 6.7388411
7: -3.0626359, 3.0674422, -3.2248030, 3.2270491, -6.2896852, 6.2922449
8: -4.3811846, 2.3532553, -4.6343346, 2.4650464, -6.8462310, 6.9875898
9: -2.9625223, 2.9923196, -3.1202610, 3.1465654, -6.1090879, 6.1125803

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0969559, upper bound: 7.0970115
time: 4.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0965950, upper bound: 7.0960264
time: 2.96 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4.8733196, 3.4150875, -3.4974098, 2.4583848, -7.3317041, 6.9124975
1: -3.3569787, 3.6769013, -2.3823414, 2.7112775, -6.0682564, 6.0592427
2: -4.8125267, 3.6287332, -3.4631758, 2.7332850, -7.5458117, 7.0919089
3: -5.8026247, 2.9310386, -4.1937175, 2.1483991, -7.9510241, 7.1247559
4: -6.0510087, 3.7373619, -4.3913722, 2.8663507, -8.9173594, 8.1287346
5: -5.0935717, 3.1353307, -3.6676021, 2.3352885, -7.4288602, 6.8029327
6: -5.9081345, 3.4266844, -4.2371554, 2.6555533, -8.5636883, 7.6638398
7: -4.3930922, 4.4052773, -3.2486572, 3.2521133, -7.6452055, 7.6539345
8: -6.4731688, 3.3423104, -4.6725078, 2.4823573, -8.9555264, 8.0148182
9: -4.2548380, 4.2879987, -3.1434884, 3.1699004, -7.4247384, 7.4314871

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0969138, upper bound: 7.0962583
time: 3.19 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0959556, upper bound: 7.0959556
time: 2.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.34
Output dim: 6, lower bound: -7.0969559, upper bound: 7.0970115
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.34
Output dim: 6, lower bound: -7.0965950, upper bound: 7.0960264
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 7.34
Output dim: 6, lower bound: -7.0969138, upper bound: 7.0962583
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 7.34
Output dim: 6, lower bound: -7.0959556, upper bound: 7.0959556

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.6329334, 1.8298770, -2.4374216, 1.6944184, -4.3273516, 4.2672987
1: -1.8476727, 2.0509577, -1.7255884, 1.8963656, -3.7440383, 3.7765460
2: -2.5526023, 2.1614063, -2.3410089, 2.0312951, -4.5838976, 4.5024152
3: -3.1043539, 1.6533577, -2.8532121, 1.5419416, -4.6462955, 4.5065699
4: -3.2416854, 2.2746739, -2.9740505, 2.1367314, -5.3784170, 5.2487245
5: -2.7091560, 1.8341887, -2.5070779, 1.7171965, -4.4263525, 4.3412666
6: -3.0980859, 2.1604331, -2.8320494, 2.0433898, -5.1414757, 4.9924822
7: -2.4759059, 2.4889889, -2.2997186, 2.3174505, -4.7933564, 4.7887077
8: -3.4445658, 1.9238836, -3.1553645, 1.7884612, -5.2330270, 5.0792480
9: -2.3868332, 2.4241126, -2.2138367, 2.2497370, -4.6365700, 4.6379490

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909785, upper bound: 7.0920346
time: 3.87 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909511, upper bound: 7.0913220
time: 9.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.6247193, 1.2025471, -3.0387695, 2.0982449, -3.7229643, 4.2413168
1: -1.2237813, 1.2836294, -2.0926971, 2.3495228, -3.5733042, 3.3763266
2: -1.4922343, 1.5102786, -2.9678431, 2.4243987, -3.9166331, 4.4781218
3: -1.8227236, 1.0962459, -3.6301174, 1.8908437, -3.7135673, 4.7263632
4: -1.9242193, 1.5827849, -3.7997649, 2.5264001, -4.4506192, 5.3825498
5: -1.6851935, 1.2667339, -3.1754608, 2.0431499, -3.7283435, 4.4421949
6: -1.7266202, 1.6306316, -3.6016171, 2.3274426, -4.0540628, 5.2322488
7: -1.5730015, 1.6157490, -2.8545973, 2.8654311, -4.4384327, 4.4703465
8: -2.0109251, 1.2833362, -4.0084200, 2.0607142, -4.0716391, 5.2917562
9: -1.5110973, 1.5633240, -2.7443178, 2.7720802, -4.2831774, 4.3076420

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0965224, upper bound: 7.0959667
time: 3.06 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0965950, upper bound: 7.0960264
time: 4.01 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -3.7557721, 2.6258345, -2.8308058, 1.9708078, -5.7265797, 5.4566402
1: -2.5453632, 2.8805034, -1.9711063, 2.2040420, -4.7494049, 4.8516097
2: -3.6965525, 2.8912737, -2.7624261, 2.2927730, -5.9893255, 5.6536999
3: -4.4708486, 2.3073516, -3.3573825, 1.7676557, -6.2385044, 5.6647339
4: -4.6804237, 3.0077486, -3.5113227, 2.4097257, -7.0901494, 6.5190716
5: -3.9299085, 2.4714446, -2.9290972, 1.9478803, -5.8777885, 5.4005418
6: -4.5257607, 2.7689562, -3.3637719, 2.2703156, -6.7960763, 6.1327281
7: -3.4472685, 3.4619601, -2.6557803, 2.6673517, -6.1146202, 6.1177406
8: -4.9827642, 2.6125965, -3.7295177, 2.0441513, -7.0269156, 6.3421144
9: -3.3381910, 3.3594613, -2.5609798, 2.5972047, -5.9353957, 5.9204412

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0968910, upper bound: 7.0961982
time: 3.30 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0969138, upper bound: 7.0962583
time: 4.41 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -4.2930956, 2.9657671, -1.8102543, 1.3060182, -5.5991139, 4.7760215
1: -2.9488897, 3.2595813, -1.3368540, 1.4199433, -4.3688331, 4.5964355
2: -4.2323236, 3.2354758, -1.6824002, 1.6263522, -5.8586760, 4.9178762
3: -5.1458182, 2.5864897, -2.0603795, 1.1955886, -6.3414068, 4.6468692
4: -5.3859663, 3.3431387, -2.1516654, 1.7057258, -7.0916920, 5.4948044
5: -4.5162325, 2.7730088, -1.8748600, 1.3618410, -5.8780737, 4.6478691
6: -5.1867132, 2.9983330, -1.9822887, 1.7093251, -6.8960381, 4.9806218
7: -3.9263825, 3.9518852, -1.7368615, 1.7757778, -5.7021604, 5.6887465
8: -5.6979742, 2.8292141, -2.2637134, 1.3865310, -7.0845051, 5.0929275
9: -3.7876399, 3.8238206, -1.6691115, 1.7134773, -5.5011172, 5.4929323

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0959101, upper bound: 7.0958958
time: 2.78 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0959556, upper bound: 7.0959556
time: 2.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.46 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0909785, upper bound: 7.0920346
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0909511, upper bound: 7.0913220
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0965224, upper bound: 7.0959667
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0965950, upper bound: 7.0960264
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0968910, upper bound: 7.0961982
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0969138, upper bound: 7.0962583
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0959101, upper bound: 7.0958958
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.46
Output dim: 6, lower bound: -7.0959556, upper bound: 7.0959556

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2.1780624, 1.5282624, -1.8365738, 1.3207884, -3.4988508, 3.3648362
1: -1.5642920, 1.6991196, -1.3551435, 1.4380955, -3.0023875, 3.0542631
2: -2.0672221, 1.8624027, -1.7093074, 1.6425363, -3.7097583, 3.5717101
3: -2.5186007, 1.3935219, -2.0862558, 1.2067111, -3.7253118, 3.4797778
4: -2.6157384, 1.9633944, -2.1781743, 1.7319585, -4.3476968, 4.1415687
5: -2.2393608, 1.5714548, -1.8957367, 1.3791633, -3.6185241, 3.4671915
6: -2.4878302, 1.9168783, -2.0262041, 1.7433164, -4.2311468, 3.9430823
7: -2.0612307, 2.0866017, -1.7560549, 1.7911713, -3.8524020, 3.8426566
8: -2.7827325, 1.6538756, -2.2987878, 1.4352944, -4.2180271, 3.9526634
9: -1.9855567, 2.0221426, -1.6879382, 1.7336419, -3.7191987, 3.7100809

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0906167, upper bound: 7.0918861
time: 4.36 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909785, upper bound: 7.0920346
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.2181396, 1.5541117, -4.1831779, 2.8407903, -5.0589299, 5.7372894
1: -1.5892729, 1.7296337, -2.8164771, 3.2593293, -4.8486023, 4.5461106
2: -2.1100831, 1.8886464, -4.1989326, 3.1865652, -5.2966480, 6.0875788
3: -2.5700107, 1.4164490, -5.0803504, 2.5347197, -5.1047306, 6.4967995
4: -2.6707728, 1.9902003, -5.3906202, 3.3663349, -6.0371075, 7.3808203
5: -2.2809227, 1.5938903, -4.2903481, 2.7413042, -5.0222268, 5.8842382
6: -2.5415378, 1.9358729, -5.1987076, 3.0277915, -5.5693293, 7.1345806
7: -2.0981176, 2.1221044, -3.8865843, 3.8525412, -5.9506588, 6.0086889
8: -2.8411617, 1.6749575, -5.6933193, 2.8661823, -5.7073441, 7.3682766
9: -2.0212154, 2.0575356, -3.7433534, 3.7946961, -5.8159113, 5.8008890

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0906896, upper bound: 7.0910040
time: 3.36 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0909511, upper bound: 7.0913220
time: 4.74 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.2801013, 1.0149674, -2.9804454, 2.0586541, -3.3387554, 3.9954128
1: -1.0207911, 1.0327268, -2.0547502, 2.3067608, -3.3275518, 3.0874770
2: -1.1512589, 1.2975036, -2.9074771, 2.3866694, -3.5379283, 4.2049809
3: -1.3771865, 0.9123880, -3.5551558, 1.8585846, -3.2357712, 4.4675436
4: -1.5024060, 1.3517700, -3.7199905, 2.4869456, -3.9893517, 5.0717607
5: -1.3367386, 1.0917231, -3.1105971, 2.0118933, -3.3486319, 4.2023201
6: -1.2505684, 1.4984875, -3.5261452, 2.2947917, -3.5453601, 5.0246325
7: -1.2692893, 1.3255349, -2.8010190, 2.8140523, -4.0833416, 4.1265540
8: -1.5704467, 1.0932724, -3.9267478, 2.0362296, -3.6066763, 5.0200205
9: -1.2150545, 1.2972100, -2.6937726, 2.7227545, -3.9378090, 3.9909825

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8122905, upper bound: 6.8478784
time: 5.30 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8083336, upper bound: 6.8027320
time: 3.17 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.4616169, 1.1124494, -2.7994106, 1.9309099, -3.3925266, 3.9118600
1: -1.1256795, 1.1640933, -1.9432424, 2.1680040, -3.2936835, 3.1073356
2: -1.3281801, 1.4096875, -2.7167377, 2.2651331, -3.5933132, 4.1264253
3: -1.6115057, 1.0082105, -3.3265510, 1.7530380, -3.3645439, 4.3347616
4: -1.7242209, 1.4729971, -3.4749751, 2.3622220, -4.0864429, 4.9479723
5: -1.5181813, 1.1832645, -2.9060826, 1.9086307, -3.4268122, 4.0893469
6: -1.5016630, 1.5668443, -3.2834420, 2.1986599, -3.7003229, 4.8502865
7: -1.4287386, 1.4764132, -2.6377954, 2.6510425, -4.0797811, 4.1142087
8: -1.8000735, 1.1934438, -3.6671295, 1.9165614, -3.7166348, 4.8605733
9: -1.3701966, 1.4353189, -2.5350122, 2.5633037, -3.9335003, 3.9703312

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 61

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8258104, upper bound: 6.8672037
time: 2.28 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8248835, upper bound: 6.8248835
time: 3.08 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.3274648, 2.3298662, -2.7591367, 1.9192500, -5.2467146, 5.0890026
1: -2.2679222, 2.5729380, -1.9226840, 2.1502416, -4.4181638, 4.4956217
2: -3.2639732, 2.6195393, -2.6837995, 2.2445045, -5.5084777, 5.3033390
3: -3.9526012, 2.0748224, -3.2622361, 1.7276464, -5.6802473, 5.3370585
4: -4.1389914, 2.7257819, -3.4124708, 2.3603761, -6.4993677, 6.1382527
5: -3.4767673, 2.2229118, -2.8498621, 1.9080626, -5.3848300, 5.0727739
6: -3.9934993, 2.5305419, -3.2702532, 2.2274466, -6.2209458, 5.8007951
7: -3.0794640, 3.0989871, -2.5874379, 2.6050198, -5.6844835, 5.6864252
8: -4.4080653, 2.3563581, -3.6268861, 2.0038133, -6.4118786, 5.9832439
9: -2.9824395, 3.0067551, -2.4969883, 2.5347705, -5.5172100, 5.5037432

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0832132, upper bound: 7.0560302
time: 2.91 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0936466, upper bound: 7.0925108
time: 4.21 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3.5583999, 2.4877894, -2.5770757, 1.7916021, -5.3500023, 5.0648651
1: -2.4055984, 2.7389140, -1.8116000, 2.0085952, -4.4141936, 4.5505142
2: -3.4965291, 2.7644322, -2.4910507, 2.1246254, -5.6211548, 5.2554827
3: -4.2314038, 2.2001226, -3.0301743, 1.6224623, -5.8538661, 5.2302971
4: -4.4307995, 2.8778224, -3.1642089, 2.2364221, -6.6672215, 6.0420313
5: -3.7209828, 2.3562779, -2.6517487, 1.8024561, -5.5234389, 5.0080266
6: -4.2813725, 2.6582527, -3.0285783, 2.1288636, -6.4102364, 5.6868310
7: -3.2771420, 3.2926846, -2.4239368, 2.4414523, -5.7185946, 5.7166214
8: -4.7179561, 2.4943044, -3.3623047, 1.8906887, -6.6086445, 5.8566093
9: -3.1742101, 3.1949060, -2.3367949, 2.3743243, -5.5485344, 5.5317011

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0836126, upper bound: 7.0564796
time: 3.58 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0937352, upper bound: 7.0926408
time: 4.91 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -3.8595572, 2.6632416, -1.7535582, 1.2754605, -5.1350174, 4.4167995
1: -2.6309264, 2.9492779, -1.3011433, 1.3805588, -4.0114851, 4.2504210
2: -3.7946687, 2.9538250, -1.6253591, 1.5922120, -5.3868809, 4.5791841
3: -4.6206217, 2.3501670, -1.9884642, 1.1666818, -5.7873034, 4.3386312
4: -4.8403707, 3.0596688, -2.0819547, 1.6690058, -6.5093765, 5.1416235
5: -4.0601749, 2.5172186, -1.8182871, 1.3339779, -5.3941526, 4.3355055
6: -4.6451025, 2.7545578, -1.9080098, 1.6829792, -6.3280816, 4.6625676
7: -3.5540948, 3.5788286, -1.6859946, 1.7281015, -5.2821960, 5.2648230
8: -5.1198840, 2.5680346, -2.1856189, 1.3634183, -6.4833021, 4.7536535
9: -3.4294026, 3.4598522, -1.6221968, 1.6685737, -5.0979762, 5.0820489

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8122905, upper bound: 6.8590029
time: 3.06 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8083336, upper bound: 6.8027320
time: 2.45 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0945625, 2.8270729, -1.5802629, 1.1780657, -5.2726283, 4.4073358
1: -2.8026428, 3.1177788, -1.1970679, 1.2511117, -4.0537548, 4.3148465
2: -4.0310345, 3.1060419, -1.4465652, 1.4832994, -5.5143337, 4.5526071
3: -4.9039040, 2.4785900, -1.7642436, 1.0731099, -5.9770136, 4.2428336
4: -5.1358466, 3.2134633, -1.8690051, 1.5535128, -6.6893597, 5.0824685
5: -4.3074832, 2.6556311, -1.6400038, 1.2443049, -5.5517883, 4.2956347
6: -4.9398251, 2.8865364, -1.6691968, 1.6122675, -6.5520926, 4.5557332
7: -3.7549803, 3.7815454, -1.5338933, 1.5788927, -5.3338728, 5.3154387
8: -5.4331393, 2.7094753, -1.9528966, 1.2596872, -6.6928263, 4.6623716
9: -3.6231771, 3.6567628, -1.4725912, 1.5282879, -5.1514649, 5.1293540

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8258104, upper bound: 6.8788222
time: 2.15 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8248835, upper bound: 6.8248835
time: 2.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.41 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0906167, upper bound: 7.0918861
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0909785, upper bound: 7.0920346
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0906896, upper bound: 7.0910040
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0909511, upper bound: 7.0913220
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8122905, upper bound: 6.8478784
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8083336, upper bound: 6.8027320
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8258104, upper bound: 6.8672037
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8248835, upper bound: 6.8248835
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0832132, upper bound: 7.0560302
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0936466, upper bound: 7.0925108
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0836126, upper bound: 7.0564796
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -7.0937352, upper bound: 7.0926408
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8122905, upper bound: 6.8590029
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8083336, upper bound: 6.8027320
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8258104, upper bound: 6.8788222
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 6, lower bound: -6.8248835, upper bound: 6.8248835

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.8069270, 1.3052510, -1.7866555, 1.2932864, -3.1002135, 3.0919065
1: -1.3348325, 1.4211415, -1.3231938, 1.4034228, -2.7382555, 2.7443352
2: -1.6796563, 1.6259388, -1.6571511, 1.6121076, -3.2917638, 3.2830899
3: -2.0453799, 1.1919216, -2.0221004, 1.1812137, -3.2265935, 3.2140222
4: -2.1378164, 1.7156173, -2.1170123, 1.6999371, -3.8377535, 3.8326297
5: -1.8650144, 1.3679547, -1.8462956, 1.3546127, -3.2196271, 3.2142503
6: -1.9939132, 1.7342405, -1.9604051, 1.7197680, -3.7136812, 3.6946456
7: -1.7271914, 1.7655576, -1.7099483, 1.7495086, -3.4767001, 3.4755058
8: -2.2597873, 1.4381379, -2.2303524, 1.4120448, -3.6718321, 3.6684904
9: -1.6635360, 1.7109709, -1.6457469, 1.6939301, -3.3574662, 3.3567178

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7896225, upper bound: 6.7855617
time: 3.63 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7431554, upper bound: 6.7713266
time: 2.42 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -2.0016496, 1.4172330, -1.6005399, 1.1889125, -3.1905622, 3.0177729
1: -1.4545006, 1.5655565, -1.2111053, 1.2642291, -2.7187295, 2.7766619
2: -1.8809969, 1.7483091, -1.4669448, 1.4949260, -3.3759229, 3.2152538
3: -2.2927911, 1.2965198, -1.7830184, 1.0822394, -3.3750305, 3.0795381
4: -2.3753452, 1.8448536, -1.8883772, 1.5775466, -3.9528918, 3.7332308
5: -2.0604374, 1.4717230, -1.6582046, 1.2571625, -3.3175998, 3.1299276
6: -2.2538607, 1.8269910, -1.7109218, 1.6369102, -3.8907709, 3.5379128
7: -1.9016838, 1.9327919, -1.5485922, 1.5917630, -3.4934468, 3.4813843
8: -2.5329905, 1.5504425, -1.9720125, 1.3028402, -3.8358307, 3.5224550
9: -1.8312256, 1.8706717, -1.4874859, 1.5417160, -3.3729415, 3.3581576

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7650037, upper bound: 6.8190047
time: 6.07 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7641755, upper bound: 6.8004704
time: 3.63 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2.1684852, 1.5219076, -3.8109708, 2.5966892, -4.7651744, 5.3328781
1: -1.5557480, 1.6938165, -2.5829182, 2.9744642, -4.5302124, 4.2767348
2: -2.0554519, 1.8569976, -3.8029206, 2.9432259, -4.9986777, 5.6599183
3: -2.5047059, 1.3895123, -4.6016741, 2.3250022, -4.8297081, 5.9911861
4: -2.6020746, 1.9562063, -4.8805785, 3.1128039, -5.7148786, 6.8367848
5: -2.2312930, 1.5664803, -3.9083443, 2.5280769, -4.7593699, 5.4748244
6: -2.4762352, 1.9075913, -4.7059293, 2.8275332, -5.3037682, 6.6135206
7: -2.0507445, 2.0799975, -3.5483589, 3.5275769, -5.5783215, 5.6283565
8: -2.7707956, 1.6502986, -5.1542797, 2.6463521, -5.4171476, 6.8045783
9: -1.9775203, 2.0151830, -3.4183702, 3.4668911, -5.4444113, 5.4335532

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7300881, upper bound: 6.7789623
time: 4.84 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7291538, upper bound: 6.7601643
time: 2.90 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1.9714727, 1.3993955, -3.9989493, 2.7196419, -4.6911144, 5.3983450
1: -1.4356196, 1.5430157, -2.7008433, 3.1184039, -4.5540237, 4.2438593
2: -1.8494481, 1.7295020, -4.0024967, 3.0660901, -4.9155383, 5.7319984
3: -2.2541807, 1.2804211, -4.8430338, 2.4307659, -4.6849465, 6.1234550
4: -2.3384259, 1.8243209, -5.1378875, 3.2407644, -5.5791903, 6.9622083
5: -2.0306373, 1.4550468, -4.1010246, 2.6359713, -4.6666088, 5.5560713
6: -2.2144637, 1.8105160, -4.9558959, 2.9295201, -5.1439838, 6.7664118
7: -1.8747034, 1.9070213, -3.7188978, 3.6915329, -5.5662365, 5.6259193
8: -2.4915376, 1.5308406, -5.4264212, 2.7580347, -5.2495723, 6.9572620
9: -1.8053051, 1.8463165, -3.5822115, 3.6322484, -5.4375534, 5.4285278

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8206997, upper bound: 6.8066632
time: 3.70 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7641336, upper bound: 6.7949144
time: 2.42 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -1.1159037, 0.9256289, -2.6188138, 1.8065407, -2.9224443, 3.5444427
1: -0.9252369, 0.9125872, -1.8289312, 2.0316362, -2.9568732, 2.7415185
2: -0.9936315, 1.1919644, -2.5284710, 2.1438887, -3.1375203, 3.7204354
3: -1.1642597, 0.8233671, -3.1020489, 1.6493915, -2.8136511, 3.9254160
4: -1.3042221, 1.2351943, -3.2272444, 2.2299054, -3.5341275, 4.4624386
5: -1.1728964, 1.0049587, -2.7093146, 1.8021253, -2.9750218, 3.7142735
6: -1.0087931, 1.4384092, -3.0291634, 2.0926032, -3.1013963, 4.4675727
7: -1.1333451, 1.1856169, -2.4736578, 2.4897101, -3.6230552, 3.6592746
8: -1.3628649, 0.9921060, -3.4121022, 1.7909365, -3.1538014, 4.4042082
9: -1.0786219, 1.1699555, -2.3805413, 2.4066377, -3.4852595, 3.5504968

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7457763, upper bound: 6.7843197
time: 3.83 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7436586, upper bound: 6.7722368
time: 2.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.6049182, 0.6630526, -5.4828448, 3.8385077, -4.4434257, 6.1458974
1: -0.6167225, 0.6133013, -3.5833688, 4.2350206, -4.8517432, 4.1966701
2: -0.5929043, 0.8318067, -5.5457053, 4.0575733, -4.6504774, 6.3775120
3: -0.5712118, 0.5625861, -6.7532806, 3.3322144, -3.9034262, 7.3158665
4: -0.7171603, 0.8748975, -7.1057138, 4.1433396, -4.8604999, 7.9806113
5: -0.7251607, 0.7255508, -6.0005341, 3.4005737, -4.1257343, 6.7260847
6: -0.3205451, 1.2966923, -6.8166928, 3.5231032, -3.8436484, 8.1133852
7: -0.7352926, 0.7656863, -5.0755005, 5.1059799, -5.8412724, 5.8411865
8: -0.7519238, 0.7412639, -7.5115399, 3.4321008, -4.1840248, 8.2528038
9: -0.6775856, 0.7772125, -4.9147186, 4.9302125, -5.6077981, 5.6919312

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7832026, upper bound: 6.7739129
time: 14.09 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8083336, upper bound: 6.8027320
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -1.2888142, 1.0194275, -2.4414253, 1.6891437, -2.9779577, 3.4608529
1: -1.0263159, 1.0380878, -1.7196634, 1.8957944, -2.9221103, 2.7577512
2: -1.1598251, 1.3016255, -2.3408637, 2.0273385, -3.1871636, 3.6424892
3: -1.3902236, 0.9164300, -2.8762331, 1.5477087, -2.9379325, 3.7926631
4: -1.5157954, 1.3520460, -2.9840264, 2.1085634, -3.6243587, 4.3360724
5: -1.3464838, 1.0928519, -2.5232153, 1.7018375, -3.0483212, 3.6160672
6: -1.2523999, 1.4960312, -2.7938118, 1.9978917, -3.2502916, 4.2898431
7: -1.2764831, 1.3317864, -2.3130813, 2.3326106, -3.6090937, 3.6448677
8: -1.5838244, 1.0824475, -3.1553731, 1.6874019, -3.2712264, 4.2378206
9: -1.2226033, 1.3040158, -2.2253261, 2.2498505, -3.4724538, 3.5293417

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8108016
time: 2.34 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8015373
time: 2.95 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.7101133, 0.7260021, -5.2657309, 3.6768260, -4.3869390, 5.9917331
1: -0.6879181, 0.6752096, -3.4431872, 4.0799284, -4.7678466, 4.1183968
2: -0.6740622, 0.9172935, -5.3167601, 3.9148264, -4.5888886, 6.2340536
3: -0.6730130, 0.6233206, -6.4714732, 3.2144499, -3.8874629, 7.0947938
4: -0.8326064, 0.9499105, -6.8134685, 3.9974923, -4.8300986, 7.7633791
5: -0.8131432, 0.7946246, -5.7552266, 3.2858739, -4.0990171, 6.5498514
6: -0.4557703, 1.3211175, -6.5400229, 3.4090168, -3.8647871, 7.8611403
7: -0.8138174, 0.8657327, -4.8788929, 4.9170179, -5.7308354, 5.7446256
8: -0.8785024, 0.7975656, -7.2105722, 3.3082287, -4.1867313, 8.0081377
9: -0.7572314, 0.8723735, -4.7134562, 4.7431235, -5.5003548, 5.5858297

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8135905, upper bound: 6.8081614
time: 3.09 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8248835, upper bound: 6.8248835
time: 3.11 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1.7920603, 1.2907751, -0.9375835, 0.8311089, -2.6231692, 2.2283585
1: -1.3302102, 1.3876505, -0.8196792, 0.7784745, -2.1086848, 2.2073298
2: -1.6591874, 1.5963695, -0.8390676, 1.0547020, -2.7138896, 2.4354372
3: -2.0482454, 1.1743820, -0.9437420, 0.7140266, -2.7622719, 2.1181240
4: -2.1440315, 1.6386340, -1.0898421, 1.0565689, -3.2006004, 2.7284760
5: -1.8568486, 1.3300834, -0.9979764, 0.8977262, -2.7545748, 2.3280597
6: -1.8625172, 1.6989169, -0.6615208, 1.3896124, -3.2521296, 2.3604379
7: -1.7128685, 1.7450342, -0.9782614, 1.0220454, -2.7349138, 2.7232957
8: -2.2248092, 1.3082533, -1.1212177, 0.8770494, -3.1018586, 2.4294710
9: -1.6429139, 1.6894416, -0.9176230, 1.0257856, -2.6686995, 2.6070647

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0135623, upper bound: 7.0094166
time: 4.16 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0107864, upper bound: 6.9734072
time: 3.95 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.5723529, 1.7782960, -1.8924066, 1.3470702, -3.9194231, 3.6707025
1: -1.8090570, 1.9867899, -1.3894187, 1.4706674, -3.2797244, 3.3762088
2: -2.4799154, 2.1098833, -1.7645113, 1.6667229, -4.1466384, 3.8743947
3: -3.0402904, 1.6193444, -2.1763296, 1.2332494, -4.2735395, 3.7956738
4: -3.1557333, 2.1957178, -2.2629006, 1.7338314, -4.8895645, 4.4586182
5: -2.6613934, 1.7723919, -1.9568522, 1.3936388, -4.0550323, 3.7292442
6: -2.9697244, 2.0767174, -2.0395617, 1.7434485, -4.7131729, 4.1162791
7: -2.4247892, 2.4422002, -1.8110381, 1.8404875, -4.2652769, 4.2532382
8: -3.3411324, 1.7945257, -2.3647022, 1.3833163, -4.7244487, 4.1592278
9: -2.3317714, 2.3644893, -1.7356150, 1.7723955, -4.1041670, 4.1001043

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8108964, upper bound: 6.8331871
time: 3.27 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7973613, upper bound: 6.7675662
time: 2.48 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.9797652, 1.3946252, -0.8221266, 0.7744747, -2.7542400, 2.2167518
1: -1.4443858, 1.5249162, -0.7513381, 0.7157390, -2.1601248, 2.2762542
2: -1.8517845, 1.7103571, -0.7514802, 0.9776247, -2.8294091, 2.4618373
3: -2.2875452, 1.2748222, -0.7998151, 0.6616318, -2.9491770, 2.0746374
4: -2.3717763, 1.7650311, -0.9554008, 0.9834064, -3.3551826, 2.7204318
5: -2.0467174, 1.4270653, -0.8966754, 0.8402441, -2.8869615, 2.3237407
6: -2.1148167, 1.7814636, -0.5163230, 1.3607087, -3.4755254, 2.2977867
7: -1.8820941, 1.9075363, -0.8879977, 0.9340284, -2.8161225, 2.7955341
8: -2.4726017, 1.4057019, -0.9877002, 0.8253828, -3.2979844, 2.3934021
9: -1.8056257, 1.8373810, -0.8264532, 0.9464258, -2.7520514, 2.6638341

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0252396, upper bound: 6.9797051
time: 2.94 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0166256, upper bound: 6.9796446
time: 4.36 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.7819679, 1.9220750, -1.6819295, 1.2290869, -4.0110550, 3.6040044
1: -1.9397700, 2.1452713, -1.2616849, 1.3140295, -3.2537994, 3.4069562
2: -2.7000840, 2.2483902, -1.5449603, 1.5367990, -4.2368832, 3.7933505
3: -3.3050694, 1.7395270, -1.9051986, 1.1191932, -4.4242625, 3.6447256
4: -3.4404488, 2.3402188, -2.0054111, 1.5918168, -5.0322657, 4.3456297
5: -2.8911791, 1.8909138, -1.7390935, 1.2840830, -4.1752620, 3.6300073
6: -3.2494447, 2.1899848, -1.7482591, 1.6549928, -4.9044375, 3.9382439
7: -2.6143942, 2.6286867, -1.6219528, 1.6559397, -4.2703338, 4.2506394
8: -3.6404257, 1.9211847, -2.0878592, 1.2571818, -4.8976073, 4.0090437
9: -2.5136523, 2.5476904, -1.5520318, 1.6038635, -4.1175156, 4.0997219

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8350123, upper bound: 6.8544501
time: 2.78 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8229823, upper bound: 6.7919789
time: 3.29 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -3.6511095, 2.5150537, -1.4101605, 1.0851723, -4.7362819, 3.9252143
1: -2.4800472, 2.7960825, -1.0948037, 1.1268893, -3.6069365, 3.8908863
2: -3.5858147, 2.8182199, -1.2783830, 1.3767473, -4.9625621, 4.0966029
3: -4.3695140, 2.2362587, -1.5472453, 0.9812655, -5.3507795, 3.7835040
4: -4.5773377, 2.9198580, -1.6648499, 1.4273497, -6.0046873, 4.5847077
5: -3.8426137, 2.3899643, -1.4701532, 1.1517423, -4.9943562, 3.8601174
6: -4.3782048, 2.6289253, -1.4149795, 1.5337218, -5.9119267, 4.0439048
7: -3.3779817, 3.3990860, -1.3816912, 1.4336621, -4.8116436, 4.7807770
8: -4.8407097, 2.4260032, -1.7367446, 1.1456786, -5.9863882, 4.1627479
9: -3.2574825, 3.2870297, -1.3271391, 1.3965997, -4.6540823, 4.6141691

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7566436, upper bound: 6.7860628
time: 3.25 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7436677, upper bound: 6.7858078
time: 3.98 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -2.9104228, 2.0195811, -4.1923742, 2.6464574, -5.5568800, 6.2119551
1: -2.0044618, 2.2602429, -2.7690411, 3.1836581, -5.1881199, 5.0292840
2: -2.8402748, 2.3397620, -4.1384683, 3.1250079, -5.9652824, 6.4782305
3: -3.4788358, 1.8264492, -5.1105819, 2.4812975, -5.9601336, 6.9370308
4: -3.6137009, 2.4120519, -5.0723581, 3.1997912, -6.8134918, 7.4844103
5: -3.0598166, 1.9609383, -4.3356981, 2.5713711, -5.6311874, 6.2966366
6: -3.4080524, 2.2169189, -5.1413221, 2.6978605, -6.1059132, 7.3582411
7: -2.7364039, 2.7623847, -3.8636389, 3.8599794, -6.5963831, 6.6260233
8: -3.8374953, 1.9554873, -5.6304932, 2.5788817, -6.4163771, 7.5859804
9: -2.6423120, 2.6687262, -3.7123504, 3.6877458, -6.3300581, 6.3810768

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7482172, upper bound: 6.7285151
time: 2.15 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7371169, upper bound: 6.7284173
time: 2.14 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -3.8846440, 2.6758442, -1.2454835, 0.9960049, -4.8806491, 3.9213276
1: -2.6509132, 2.9636590, -1.0012426, 1.0058749, -3.6567881, 3.9649017
2: -3.8190279, 2.9697664, -1.1165118, 1.2736193, -5.0926471, 4.0862780
3: -4.6507788, 2.3639231, -1.3321977, 0.8934977, -5.5442762, 3.6961207
4: -4.8717680, 3.0730743, -1.4619317, 1.3182974, -6.1900654, 4.5350060
5: -4.0879722, 2.5271139, -1.3042754, 1.0690455, -5.1570177, 3.8313894
6: -4.6717987, 2.7577848, -1.1890392, 1.4767330, -6.1485319, 3.9468241
7: -3.5772541, 3.6008515, -1.2371770, 1.2952344, -4.8724885, 4.8380284
8: -5.1516790, 2.5669439, -1.5269322, 1.0571166, -6.2087955, 4.0938759
9: -3.4499736, 3.4819593, -1.1851259, 1.2707101, -4.7206836, 4.6670852

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8225903
time: 2.03 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8130231
time: 3.02 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -3.1242878, 2.1684322, -3.9114344, 2.4932551, -5.6175432, 6.0798664
1: -2.1371844, 2.4213591, -2.5985751, 2.9825912, -5.1197758, 5.0199342
2: -3.0630763, 2.4816010, -3.8446090, 2.9212246, -5.9843006, 6.3262100
3: -3.7477884, 1.9500444, -4.7969375, 2.3487206, -6.0965090, 6.7469816
4: -3.9025047, 2.5582409, -4.7565026, 3.0571356, -6.9596405, 7.3147435
5: -3.2990062, 2.0805409, -4.0844216, 2.4098310, -5.7088375, 6.1649628
6: -3.6919923, 2.3311329, -4.8228912, 2.5243742, -6.2163668, 7.1540241
7: -2.9291940, 2.9530916, -3.5813894, 3.6237810, -6.5529747, 6.5344810
8: -4.1401482, 2.0815663, -5.0020580, 2.4568057, -6.5969539, 7.0836244
9: -2.8283019, 2.8541131, -3.5073419, 3.3930933, -6.2213955, 6.3614550

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7837492, upper bound: 6.7635137
time: 2.04 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7634111, upper bound: 6.7634111
time: 2.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.35 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7896225, upper bound: 6.7855617
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7431554, upper bound: 6.7713266
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7650037, upper bound: 6.8190047
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7641755, upper bound: 6.8004704
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7300881, upper bound: 6.7789623
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7291538, upper bound: 6.7601643
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8206997, upper bound: 6.8066632
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7641336, upper bound: 6.7949144
IS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7457763, upper bound: 6.7843197
IS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7436586, upper bound: 6.7722368
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7832026, upper bound: 6.7739129
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8083336, upper bound: 6.8027320
IS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8108016
IS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8015373
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8135905, upper bound: 6.8081614
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8248835, upper bound: 6.8248835
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -7.0135623, upper bound: 7.0094166
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -7.0107864, upper bound: 6.9734072
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8108964, upper bound: 6.8331871
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7973613, upper bound: 6.7675662
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -7.0252396, upper bound: 6.9797051
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -7.0166256, upper bound: 6.9796446
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8350123, upper bound: 6.8544501
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.8229823, upper bound: 6.7919789
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7566436, upper bound: 6.7860628
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7436677, upper bound: 6.7858078
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7482172, upper bound: 6.7285151
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7371169, upper bound: 6.7284173
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8225903
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8130231
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7837492, upper bound: 6.7635137
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.35
Output dim: 6, lower bound: -6.7634111, upper bound: 6.7634111

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1.4656312, 1.1177237, -1.6077275, 1.1942365, -2.6598678, 2.7254512
1: -1.1299505, 1.1680810, -1.2150478, 1.2706398, -2.4005904, 2.3831289
2: -1.3370554, 1.4120853, -1.4760542, 1.4994338, -2.8364892, 2.8881395
3: -1.6129632, 1.0117726, -1.7962706, 1.0869708, -2.6999340, 2.8080432
4: -1.7267547, 1.4821599, -1.9004939, 1.5784059, -3.3051605, 3.3826537
5: -1.5275483, 1.1849457, -1.6691661, 1.2585977, -2.7861462, 2.8541117
6: -1.5190735, 1.5690877, -1.7121286, 1.6299273, -3.1490006, 3.2812164
7: -1.4305096, 1.4794847, -1.5555849, 1.5989933, -3.0295029, 3.0350695
8: -1.8022414, 1.2165834, -1.9847760, 1.2954109, -3.0976524, 3.2013593
9: -1.3764659, 1.4395196, -1.4960026, 1.5495313, -2.9259973, 2.9355221

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5893154, upper bound: 6.5743473
time: 3.11 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7896225, upper bound: 6.7855617
time: 3.37 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.2042456, 2.6735210, -0.9673557, 0.8538589, -5.0581045, 3.6408768
1: -2.7776265, 3.1941123, -0.8387876, 0.8151455, -3.5927720, 4.0328999
2: -4.1495981, 3.1450424, -0.8664656, 1.0923233, -5.2419214, 4.0115080
3: -5.1156988, 2.4896729, -0.9692398, 0.7454354, -5.8611341, 3.4589128
4: -5.0818119, 3.2524562, -1.1200403, 1.1276932, -6.2095051, 4.3724966
5: -4.3440161, 2.5837431, -1.0338033, 0.9277864, -5.2718024, 3.6175466
6: -5.1703911, 2.7261398, -0.7939550, 1.3865098, -6.5569010, 3.5200949
7: -3.8972759, 3.8663220, -1.0100968, 1.0636472, -4.9609232, 4.8764191
8: -5.6498480, 2.6624975, -1.1707456, 0.9315914, -6.5814395, 3.8332431
9: -3.7205167, 3.7109706, -0.9568265, 1.0577475, -4.7782640, 4.6677971

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5733636, upper bound: 6.5873418
time: 2.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7069835, upper bound: 6.7351182
time: 2.32 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1.8180445, 1.3125107, -1.2742679, 1.0129293, -2.8309739, 2.5867786
1: -1.3429316, 1.4268587, -1.0202945, 1.0263485, -2.3692801, 2.4471531
2: -1.6926923, 1.6327437, -1.1469513, 1.2920940, -2.9847863, 2.7796950
3: -2.0621243, 1.1984694, -1.3673289, 0.9086646, -2.9707890, 2.5657983
4: -2.1547523, 1.7195914, -1.4956639, 1.3498957, -3.5046480, 3.2152553
5: -1.8800108, 1.3691524, -1.3339361, 1.0863544, -2.9663653, 2.7030885
6: -2.0033829, 1.7296242, -1.2442305, 1.4986197, -3.5020027, 2.9738545
7: -1.7391281, 1.7761884, -1.2621777, 1.3182151, -3.0573432, 3.0383661
8: -2.2779751, 1.4285830, -1.5632944, 1.0908802, -3.3688552, 2.9918776
9: -1.6747458, 1.7216185, -1.2104734, 1.2929065, -2.9676523, 2.9320920

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6888792, upper bound: 6.7281810
time: 3.08 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7650037, upper bound: 6.8190047
time: 2.80 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.1406977, 0.9434780, -3.9804139, 2.5087118, -3.6494095, 4.9238920
1: -0.9399607, 0.9316579, -2.6249130, 3.0291750, -3.9691358, 3.5565710
2: -1.0204525, 1.2065253, -3.8850689, 2.9598939, -3.9803464, 5.0915942
3: -1.1926682, 0.8387436, -4.8577332, 2.3629003, -3.5555685, 5.6964769
4: -1.3326808, 1.2462949, -4.8068871, 3.0836613, -4.4163423, 6.0531821
5: -1.2049520, 1.0155376, -4.1137738, 2.4353523, -3.6403043, 5.1293116
6: -1.0479896, 1.4357522, -4.8702970, 2.5850098, -3.6329994, 6.3060493
7: -1.1476610, 1.2051332, -3.6154680, 3.6458449, -4.7935057, 4.8206015
8: -1.3874919, 1.0279082, -5.0795789, 2.4948781, -3.8823700, 6.1074872
9: -1.1017811, 1.1910899, -3.5319026, 3.4410717, -4.5428529, 4.7229924

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5902119, upper bound: 6.6349693
time: 2.14 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7289493, upper bound: 6.7648816
time: 5.98 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.9829353, 1.4079945, -3.4533782, 2.3628924, -4.3458276, 4.8613729
1: -1.4423466, 1.5517573, -2.3598509, 2.6965203, -4.1388669, 3.9116082
2: -1.8633758, 1.7376819, -3.4240661, 2.7083657, -4.5717416, 5.1617479
3: -2.2718658, 1.2886890, -4.1457973, 2.1235058, -4.3953714, 5.4344864
4: -2.3561244, 1.8285536, -4.3915753, 2.8623929, -5.2185173, 6.2201290
5: -2.0477588, 1.4577075, -3.5466523, 2.3148780, -4.3626366, 5.0043597
6: -2.2233388, 1.8034675, -4.2228732, 2.6184762, -4.8418150, 6.0263405
7: -1.8864284, 1.9196322, -3.2259603, 3.2162147, -5.1026430, 5.1455927
8: -2.5125551, 1.5259860, -4.6381197, 2.4055486, -4.9181037, 6.1641054
9: -1.8185169, 1.8589997, -3.1086586, 3.1530554, -4.9715724, 4.9676580

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5870661, upper bound: 6.6395115
time: 2.52 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7300881, upper bound: 6.7789623
time: 3.62 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.2913361, 1.0258901, -6.1112700, 4.1037116, -5.3950477, 7.1371603
1: -1.0264698, 1.0428939, -3.9947546, 4.7420106, -5.7684803, 5.0376482
2: -1.1676742, 1.3025520, -6.2477674, 4.4505711, -5.6182451, 7.5503197
3: -1.3910868, 0.9207898, -7.5726643, 3.6570737, -5.0481606, 8.4934540
4: -1.5174115, 1.3477702, -8.0409527, 4.6358376, -6.1532488, 9.3887234
5: -1.3576853, 1.0920988, -6.3286738, 3.7983568, -5.1560421, 7.4207726
6: -1.2585596, 1.4836056, -7.7372828, 3.9583702, -5.2169299, 9.2208881
7: -1.2740953, 1.3339242, -5.6527262, 5.5935473, -6.8676424, 6.9866505
8: -1.5807930, 1.1090052, -8.5052948, 3.8017297, -5.3825226, 9.6142998
9: -1.2294911, 1.3081810, -5.4580836, 5.5129733, -6.7424645, 6.7662649

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5211878, upper bound: 6.5632971
time: 2.75 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6933928, upper bound: 6.7239822
time: 2.82 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -1.6215264, 1.2028878, -3.8118203, 2.5972607, -4.2187872, 5.0147080
1: -1.2231647, 1.2821181, -2.5840790, 2.9729881, -4.1961527, 3.8661971
2: -1.4923759, 1.5083447, -3.8042011, 2.9431472, -4.4355230, 5.3125458
3: -1.8135389, 1.0949357, -4.6044464, 2.3253119, -4.1388507, 5.6993818
4: -1.9161406, 1.5857561, -4.8819828, 3.1097810, -5.0259218, 6.4677391
5: -1.6850983, 1.2642974, -3.9117107, 2.5244260, -4.2095242, 5.1760082
6: -1.7323669, 1.6311113, -4.7033668, 2.8203390, -4.5527058, 6.3344779
7: -1.5675116, 1.6108340, -3.5501509, 3.5285830, -5.0960946, 5.1609850
8: -2.0048676, 1.3005822, -5.1563058, 2.6319358, -4.6368036, 6.4568882
9: -1.5098953, 1.5616140, -3.4201274, 3.4679697, -4.9778652, 4.9817414

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7267007, upper bound: 6.7052129
time: 3.26 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8206997, upper bound: 6.8066632
time: 3.81 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -4.4455557, 2.8830492, -3.0466218, 2.0965910, -6.5421467, 5.9296713
1: -2.9320860, 3.3993626, -2.1039200, 2.3783689, -5.3104548, 5.5032825
2: -4.4328604, 3.2980077, -2.9936049, 2.4415047, -6.8743649, 6.2916126
3: -5.4071960, 2.6572955, -3.6298423, 1.8963690, -7.3035650, 6.2871380
4: -5.3725882, 3.4078009, -3.8356442, 2.5696104, -7.9421988, 7.2434454
5: -4.5914826, 2.7404535, -3.1406698, 2.0649502, -6.6564331, 5.8811235
6: -5.4662728, 2.9052796, -3.6656313, 2.3627305, -7.8290033, 6.5709109
7: -4.0918655, 4.1008654, -2.8619533, 2.8655322, -6.9573975, 6.9628186
8: -5.9910774, 2.8114290, -4.0533409, 2.1045713, -8.0956488, 6.8647699
9: -3.9851902, 3.9055161, -2.7603726, 2.7977777, -6.7829676, 6.6658888

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6281823, upper bound: 6.6357824
time: 2.72 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7289095, upper bound: 6.7592087
time: 2.27 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.7774950, 0.7557729, -1.9806453, 1.4001518, -2.1776469, 2.7364182
1: -0.7267990, 0.7113059, -1.4379196, 1.5456225, -2.2724214, 2.1492255
2: -0.7225679, 0.9653759, -1.8591019, 1.7277679, -2.4503357, 2.8244777
3: -0.7457126, 0.6539863, -2.2896352, 1.2867482, -2.0324607, 2.9436216
4: -0.9032490, 1.0162886, -2.3687105, 1.7999815, -2.7032304, 3.3849993
5: -0.8639235, 0.8344902, -2.0500431, 1.4442109, -2.3081346, 2.8845334
6: -0.5592436, 1.3576204, -2.1698775, 1.7710495, -2.3302932, 3.5274978
7: -0.8689499, 0.9198765, -1.8946538, 1.9244325, -2.7933824, 2.8145304
8: -0.9611573, 0.8240232, -2.4973993, 1.4288648, -2.3900223, 3.3214226
9: -0.8067783, 0.9223107, -1.8204257, 1.8497779, -2.6565561, 2.7427363

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7324043, upper bound: 6.7692941
time: 3.11 seconds

## Relational analysis of IS_A1_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7457763, upper bound: 6.7843197
time: 3.34 seconds

## BFS IS instance: IS_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.7959147, 0.7654837, -4.2568169, 2.9555817, -3.7514963, 5.0223007
1: -0.7376432, 0.7202526, -2.8397961, 3.2954986, -4.0331416, 3.5600486
2: -0.7353044, 0.9783653, -4.2473369, 3.2521722, -3.9874766, 5.2257023
3: -0.7641405, 0.6629410, -5.1511111, 2.6105280, -3.3746686, 5.8140521
4: -0.9221241, 1.0272777, -5.4851322, 3.3910170, -4.3131409, 6.5124102
5: -0.8787765, 0.8442978, -4.5587969, 2.7395241, -3.6183007, 5.4030948
6: -0.5814804, 1.3610289, -5.2828326, 2.9842491, -3.5657296, 6.6438618
7: -0.8817301, 0.9342685, -3.9654384, 3.9779906, -4.8597207, 4.8997068
8: -0.9800916, 0.8342926, -5.7671781, 2.7782593, -3.7583508, 6.6014705
9: -0.8202018, 0.9362476, -3.8073287, 3.8557291, -4.6759310, 4.7435765

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7298808, upper bound: 6.7570433
time: 3.43 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7436586, upper bound: 6.7722368
time: 3.53 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.3620359, 0.4634898, -3.5258827, 2.4590473, -2.8210833, 3.9893725
1: -0.4090886, 0.4441544, -2.3745112, 2.7350757, -3.1441643, 2.8186655
2: -0.3970938, 0.6042718, -3.4914870, 2.7456050, -3.1426988, 4.0957589
3: -0.3820429, 0.3780390, -4.2727633, 2.1908262, -2.5728691, 4.6508021
4: -0.4443849, 0.6387508, -4.4366570, 2.7935855, -3.2379704, 5.0754080
5: -0.4974068, 0.5308236, -3.7801375, 2.2949953, -2.7924023, 4.3109612
6: 0.0348080, 1.2436503, -4.1799307, 2.4991210, -2.4643130, 5.4235811
7: -0.5415213, 0.4894830, -3.2936287, 3.3273325, -3.8688538, 3.7831118
8: -0.4821790, 0.5462906, -4.7113476, 2.2960372, -2.7782164, 5.2576380
9: -0.4638316, 0.4908910, -3.1909838, 3.2084999, -3.6723316, 3.6818748

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7199241, upper bound: 6.7289159
time: 2.20 seconds

## Relational analysis of IS_A1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7174622, upper bound: 6.7051404
time: 2.26 seconds

## BFS IS instance: IS_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.4551817, 0.5553178, -4.9955816, 3.4960506, -3.9512322, 5.5508995
1: -0.4932153, 0.5160185, -3.2820578, 3.8623583, -4.3555737, 3.7980762
2: -0.4784333, 0.7032356, -5.0341983, 3.7317374, -4.2101707, 5.7374339
3: -0.4485444, 0.4567637, -6.1386437, 3.0488696, -3.4974139, 6.5954075
4: -0.5483559, 0.7459837, -6.4434528, 3.8079212, -4.3562770, 7.1894364
5: -0.5957980, 0.6167583, -5.4500122, 3.1257038, -3.7215018, 6.0667706
6: -0.1105341, 1.2643772, -6.1614237, 3.2633300, -3.3738642, 7.4258008
7: -0.6261765, 0.6087633, -4.6335645, 4.6649346, -5.2911110, 5.2423277
8: -0.5830075, 0.6416426, -6.8155622, 3.1444492, -3.7274566, 7.4572048
9: -0.5520564, 0.6218908, -4.4864893, 4.5027785, -5.0548348, 5.1083803

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7576588, upper bound: 6.7726793
time: 2.19 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7567487, upper bound: 6.7497386
time: 3.32 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.9188898, 0.8262801, -1.8058016, 1.3002639, -2.2191536, 2.6320817
1: -0.8107442, 0.7849189, -1.3329517, 1.4146345, -2.2253785, 2.1178706
2: -0.8256814, 1.0607946, -1.6773698, 1.6182524, -2.4439340, 2.7381644
3: -0.9099758, 0.7188944, -2.0660379, 1.1910834, -2.1010592, 2.7849321
4: -1.0592141, 1.1076291, -2.1559641, 1.6825272, -2.7417412, 3.2635932
5: -0.9845543, 0.9060561, -1.8708652, 1.3523440, -2.3368983, 2.7769213
6: -0.7357167, 1.3901290, -1.9351935, 1.6943932, -2.4301100, 3.3253226
7: -0.9757005, 1.0275997, -1.7367060, 1.7722075, -2.7479081, 2.7643056
8: -1.1174501, 0.8917103, -2.2566161, 1.3323976, -2.4498477, 3.1483264
9: -0.9142087, 1.0217314, -1.6661229, 1.7070698, -2.6212785, 2.6878543

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6795745, upper bound: 6.7141418
time: 3.29 seconds

## Relational analysis of IS_A1_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8108016
time: 2.26 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.9447896, 0.8388870, -4.0583520, 2.7468629, -3.6916525, 4.8972387
1: -0.8259594, 0.8001931, -2.7243474, 3.1416938, -3.9676533, 3.5245404
2: -0.8470337, 1.0774567, -4.0496874, 3.0977089, -3.9447427, 5.1271439
3: -0.9425737, 0.7316889, -4.9140496, 2.4794574, -3.4220309, 5.6457386
4: -1.0910131, 1.1230700, -5.2289672, 3.2551496, -4.3461628, 6.3520374
5: -1.0081631, 0.9188948, -4.1860075, 2.6313291, -3.6394920, 5.1049023
6: -0.7702932, 1.3964214, -5.0194101, 2.8877983, -3.6580915, 6.4158316
7: -0.9962261, 1.0472881, -3.7822800, 3.7711155, -4.7673416, 4.8295679
8: -1.1493261, 0.9053925, -5.5028477, 2.6405704, -3.7898965, 6.4082403
9: -0.9355295, 1.0396693, -3.6400261, 3.6827583, -4.6182880, 4.6796951

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6336451, upper bound: 6.6735222
time: 3.08 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8015373
time: 2.76 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.4020162, 0.5040161, -3.4236963, 2.3844404, -2.7864566, 3.9277124
1: -0.4454182, 0.4737764, -2.3095570, 2.6597800, -3.1051984, 2.7833333
2: -0.4358574, 0.6460552, -3.3837538, 2.6774340, -3.1132913, 4.0298090
3: -0.4133341, 0.4135066, -4.1401758, 2.1341181, -2.5474522, 4.5536823
4: -0.4875953, 0.6888335, -4.2982736, 2.7225068, -3.2101021, 4.9871073
5: -0.5403395, 0.5702986, -3.6647716, 2.2385221, -2.7788615, 4.2350702
6: -0.0285799, 1.2556812, -4.0458622, 2.4452014, -2.4737813, 5.3015432
7: -0.5809784, 0.5411430, -3.2007012, 3.2373867, -3.8183651, 3.7418442
8: -0.5247505, 0.5922620, -4.5657301, 2.2355573, -2.7603078, 5.1579924
9: -0.5002365, 0.5507379, -3.0980225, 3.1188385, -3.6190751, 3.6487603

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7568891, upper bound: 6.7709936
time: 2.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7568336, upper bound: 6.7529335
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.5267831, 0.6081925, -4.7742367, 3.3329890, -3.8597722, 5.3824291
1: -0.5553352, 0.5631958, -3.1406312, 3.7017770, -4.2571120, 3.7038269
2: -0.5340430, 0.7671233, -4.8008933, 3.5856295, -4.1196723, 5.5680165
3: -0.5022240, 0.5092545, -5.8527198, 2.9269478, -3.4291718, 6.3619742
4: -0.6310920, 0.8091508, -6.1450577, 3.6589000, -4.2899919, 6.9542084
5: -0.6596137, 0.6690100, -5.1996861, 3.0071807, -3.6667943, 5.8686962
6: -0.2124596, 1.2818182, -5.8768840, 3.1469984, -3.3594580, 7.1587019
7: -0.6793203, 0.6852384, -4.4331045, 4.4707623, -5.1500826, 5.1183429
8: -0.6628952, 0.6930415, -6.5065784, 3.0169013, -3.6797965, 7.1996198
9: -0.6153820, 0.6976116, -4.2834244, 4.3111472, -4.9265294, 4.9810362

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7700792, upper bound: 6.7873137
time: 1.87 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7698538, upper bound: 6.7698538
time: 2.71 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.4369901, 1.0948048, -0.6039566, 0.6520834, -2.0890737, 1.6987613
1: -1.1157733, 1.1271558, -0.6127001, 0.5964162, -1.7121894, 1.7398559
2: -1.2990441, 1.3769622, -0.5927612, 0.8096858, -2.1087298, 1.9697235
3: -1.5873564, 0.9827821, -0.5808723, 0.5430241, -2.1303806, 1.5636544
4: -1.7060670, 1.3989789, -0.7202127, 0.8347145, -2.5407815, 2.1191916
5: -1.4927803, 1.1484952, -0.7148567, 0.7095782, -2.2023585, 1.8633519
6: -1.3710593, 1.5549684, -0.2427849, 1.3168560, -2.6879153, 1.7977532
7: -1.3959146, 1.4378814, -0.7273765, 0.7417650, -2.1376796, 2.1652579
8: -1.7579520, 1.1245754, -0.7383535, 0.7182898, -2.4762418, 1.8629289
9: -1.3354666, 1.4090850, -0.6623527, 0.7627265, -2.0981932, 2.0714378

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8900454, upper bound: 6.8739922
time: 12.32 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8643366, upper bound: 6.8697334
time: 3.75 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -1.4393867, 1.0959585, -2.3287468, 1.5206637, -2.9600506, 3.4247053
1: -1.1172655, 1.1284889, -1.6446337, 1.6430289, -2.7602944, 2.7731225
2: -1.3011973, 1.3782222, -1.9724212, 2.0175462, -3.3187435, 3.3506434
3: -1.5902421, 0.9838022, -2.7251275, 1.4148109, -3.0050530, 3.7089295
4: -1.7088575, 1.4001584, -2.8620934, 1.9847662, -3.6936238, 4.2622519
5: -1.4948679, 1.1496414, -2.2760768, 1.6114988, -3.1063666, 3.4257183
6: -1.3739325, 1.5565041, -2.6173897, 1.7749752, -3.1489077, 4.1738939
7: -1.3978548, 1.4395641, -2.1106048, 2.1240070, -3.5218618, 3.5501690
8: -1.7608854, 1.1256914, -2.8969648, 1.5426855, -3.3035707, 4.0226564
9: -1.3372058, 1.4107126, -2.0825191, 2.0579791, -3.3951850, 3.4932318

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9329495, upper bound: 6.9070298
time: 4.02 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0107864, upper bound: 6.9734072
time: 3.73 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -2.3903136, 1.6572192, -1.5522529, 1.1580323, -3.5483460, 3.2094722
1: -1.6949623, 1.8463147, -1.1819961, 1.2183152, -2.9132776, 3.0283108
2: -2.2858288, 1.9877431, -1.4128278, 1.4548978, -3.7407265, 3.4005709
3: -2.8077962, 1.5141864, -1.7367104, 1.0492839, -3.8570800, 3.2508969
4: -2.9044552, 2.0640938, -1.8450413, 1.4950027, -4.3994579, 3.9091351
5: -2.4744191, 1.6643593, -1.6085733, 1.2145153, -3.6889343, 3.2729325
6: -2.7146740, 1.9759246, -1.5582271, 1.5962412, -4.3109150, 3.5341516
7: -2.2580824, 2.2799058, -1.5040739, 1.5426196, -3.8007021, 3.7839797
8: -3.0753329, 1.6720676, -1.9132006, 1.1892104, -4.2645435, 3.5852683
9: -2.1724238, 2.2018805, -1.4398360, 1.5012906, -3.6737144, 3.6417165

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7466009, upper bound: 6.7729710
time: 3.98 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7415226, upper bound: 6.7586519
time: 2.63 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.7180742, 1.2549865, -4.3598313, 2.7205362, -4.4386106, 5.6148176
1: -1.2822310, 1.3413743, -2.8639860, 3.3041456, -4.5863767, 4.2053604
2: -1.5874662, 1.5546310, -4.3199940, 3.1651521, -4.7526183, 5.8746252
3: -1.9487247, 1.1417638, -5.3179007, 2.5793705, -4.5280952, 6.4596643
4: -2.0498033, 1.5874770, -5.2489457, 3.3214042, -5.3712072, 6.8364229
5: -1.7912147, 1.2919154, -4.4873452, 2.6540074, -4.4452219, 5.7792606
6: -1.7778121, 1.6495084, -5.3010516, 2.7552295, -4.5330415, 6.9505601
7: -1.6434016, 1.6883447, -4.0219641, 3.9993210, -5.6427226, 5.7103090
8: -2.1269646, 1.2853900, -5.6107883, 2.6760511, -4.8030157, 6.8961782
9: -1.5849427, 1.6357132, -3.9026248, 3.7335756, -5.3185182, 5.5383382

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7306932, upper bound: 6.6934277
time: 3.38 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7260309, upper bound: 6.6933928
time: 3.18 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.4892339, 1.1225842, -0.5986571, 0.6487733, -2.1380072, 1.7212414
1: -1.1470768, 1.1634908, -0.6087823, 0.5931734, -1.7402503, 1.7722731
2: -1.3490028, 1.4084227, -0.5879359, 0.8057926, -2.1547954, 1.9963586
3: -1.6545471, 1.0099727, -0.5750998, 0.5400865, -2.1946335, 1.5850725
4: -1.7699399, 1.4341813, -0.7139294, 0.8304937, -2.6004336, 2.1481106
5: -1.5443468, 1.1748308, -0.7108055, 0.7058730, -2.2502198, 1.8856363
6: -1.4419072, 1.5766249, -0.2380057, 1.3144029, -2.7563100, 1.8146305
7: -1.4420153, 1.4803002, -0.7228244, 0.7372593, -2.1792746, 2.2031245
8: -1.8240300, 1.1509371, -0.7319410, 0.7155907, -2.5396209, 1.8828781
9: -1.3794839, 1.4485124, -0.6584458, 0.7576506, -2.1371346, 2.1069584

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.9042092, upper bound: 6.8424755
time: 2.94 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.8877906, upper bound: 6.8420275
time: 4.23 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -3.6011913, 2.3317246, -0.6043925, 0.6525266, -4.2537179, 2.9361172
1: -2.4314828, 2.7460353, -0.6131226, 0.5965956, -3.0280783, 3.3591580
2: -3.5294085, 2.7263813, -0.5928327, 0.8101498, -4.3395581, 3.3192141
3: -4.3575768, 2.1693752, -0.5804892, 0.5434901, -4.9010668, 2.7498643
4: -4.3391938, 2.9021921, -0.7204832, 0.8344610, -5.1736546, 3.6226754
5: -3.7133157, 2.2866125, -0.7154610, 0.7098240, -4.4231396, 3.0020735
6: -4.3782415, 2.4937439, -0.2449327, 1.3164440, -5.6946855, 2.7386765
7: -3.3585739, 3.3578532, -0.7272737, 0.7425320, -4.1011057, 4.0851269
8: -4.6494532, 2.2586355, -0.7390707, 0.7185869, -5.3680401, 2.9977062
9: -3.2415392, 3.1354218, -0.6625959, 0.7631403, -4.0046797, 3.7980177

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0156724, upper bound: 6.9796446
time: 4.01 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -7.0166256, upper bound: 6.9796446
time: 4.32 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2.5952814, 1.7924416, -1.3522263, 1.0493797, -3.6446609, 3.1446679
1: -1.8229785, 2.0029712, -1.0653565, 1.0722151, -2.8951936, 3.0683277
2: -2.5035083, 2.1228642, -1.2170352, 1.3311673, -3.8346758, 3.3398995
3: -3.0712419, 1.6314124, -1.4773483, 0.9418380, -4.0130796, 3.1087608
4: -3.1861610, 2.2070038, -1.5999223, 1.3607419, -4.5469027, 3.8069263
5: -2.6866477, 1.7818414, -1.4066029, 1.1126792, -3.7993269, 3.1884441
6: -2.9918056, 2.0850577, -1.2835934, 1.5200412, -4.5118465, 3.3686512
7: -2.4452949, 2.4615304, -1.3265433, 1.3747375, -3.8200324, 3.7880738
8: -3.3738861, 1.7915177, -1.6579028, 1.0836405, -4.4575267, 3.4494205
9: -2.3520434, 2.3839669, -1.2664199, 1.3471732, -3.6992166, 3.6503868

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7737444, upper bound: 6.7971220
time: 4.10 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7710483, upper bound: 6.7852799
time: 3.18 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1.9187360, 1.3660364, -4.0569658, 2.5632553, -4.4819913, 5.4230022
1: -1.4034166, 1.4890176, -2.6885812, 3.0692170, -4.4726334, 4.1775990
2: -1.7939736, 1.6783259, -3.9990335, 2.9947696, -4.7887430, 5.6773596
3: -2.2043507, 1.2495105, -4.9842300, 2.4125352, -4.6168861, 6.2337408
4: -2.2929525, 1.7223637, -4.9477530, 3.1236506, -5.4166031, 6.6701164
5: -1.9946747, 1.3959846, -4.2166948, 2.4726987, -4.4673734, 5.6126795
6: -2.0493035, 1.7373482, -4.9757748, 2.5884209, -4.6377244, 6.7131228
7: -1.8243082, 1.8617988, -3.7071633, 3.7288439, -5.5531521, 5.5689621
8: -2.3977203, 1.3934880, -5.2235131, 2.4971757, -4.8948960, 6.6170011
9: -1.7587882, 1.7962559, -3.6184144, 3.4936602, -5.2524486, 5.4146705

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7671955, upper bound: 6.7289493
time: 3.30 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7608279, upper bound: 6.7289095
time: 2.58 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.9739692, 2.0554488, -1.0206277, 0.8770651, -3.8510342, 3.0760765
1: -2.0519230, 2.2983625, -0.8700152, 0.8470704, -2.8989935, 3.1683776
2: -2.8995140, 2.3791945, -0.9099627, 1.1283083, -4.0278225, 3.2891572
3: -3.5462356, 1.8562465, -1.0403639, 0.7718893, -4.3181248, 2.8966103
4: -3.7033727, 2.4708171, -1.1873906, 1.1671934, -4.8705664, 3.6582077
5: -3.1101575, 2.0005774, -1.0803444, 0.9552569, -4.0654144, 3.0809219
6: -3.5044279, 2.2807140, -0.8684775, 1.4103112, -4.9147391, 3.1491914
7: -2.7904403, 2.8077843, -1.0557494, 1.1074283, -3.8978686, 3.8635337
8: -3.9153697, 2.0174999, -1.2431302, 0.9418388, -4.8572087, 3.2606301
9: -2.6864433, 2.7158241, -0.9985068, 1.0967877, -3.7832310, 3.7143309

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4843250, upper bound: 6.5236738
time: 3.16 seconds

## Relational analysis of IS_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7566436, upper bound: 6.7860628
time: 3.22 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.4304471, 3.7092505, -1.0632467, 0.8988095, -6.3292565, 4.7724972
1: -3.7625937, 4.0932751, -0.8951869, 0.8744538, -4.6370478, 4.9884620
2: -5.3349452, 3.9776304, -0.9472702, 1.1565087, -6.4914541, 4.9249005
3: -6.4446898, 3.2261081, -1.0956883, 0.7950953, -7.2397852, 4.3217964
4: -6.8118844, 4.1060367, -1.2402575, 1.1945527, -8.0064373, 5.3462944
5: -5.6778598, 3.4529762, -1.1219908, 0.9764714, -6.6543312, 4.5749669
6: -6.6502419, 3.5970635, -0.9278635, 1.4215541, -8.0717964, 4.5249271
7: -4.8707237, 4.9326038, -1.0894899, 1.1415820, -6.0123057, 6.0220938
8: -7.1874075, 3.5578034, -1.2960876, 0.9646993, -8.1521072, 4.8538909
9: -4.7084846, 4.7609005, -1.0344578, 1.1291745, -5.8376589, 5.7953582

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5562797, upper bound: 6.6039976
time: 3.31 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7073368, upper bound: 6.7494012
time: 2.03 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2.2883945, 1.5936680, -3.7424512, 2.3963182, -4.6847124, 5.3361192
1: -1.6233672, 1.7785687, -2.4961481, 2.8542256, -4.4775929, 4.2747169
2: -2.1837459, 1.9232831, -3.6774147, 2.8425734, -5.0263195, 5.6006975
3: -2.6810558, 1.4628005, -4.5413547, 2.2416966, -4.9227524, 6.0041552
4: -2.7616897, 1.9803536, -4.5241051, 2.9060788, -5.6677685, 6.5044584
5: -2.3756907, 1.6049790, -3.8792884, 2.3402042, -4.7158947, 5.4842672
6: -2.5635233, 1.9032965, -4.5346231, 2.4926639, -5.0561872, 6.4379196
7: -2.1667941, 2.1979365, -3.4632556, 3.4718199, -5.6386137, 5.6611919
8: -2.9332492, 1.6001210, -4.9973516, 2.3380511, -5.2713003, 6.5974727
9: -2.0921075, 2.1160007, -3.3302732, 3.3183515, -5.4104590, 5.4462738

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5562544, upper bound: 6.5625495
time: 2.66 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7124061, upper bound: 6.6927644
time: 2.28 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4982204, 3.1333730, -3.8030729, 2.4305277, -6.9287481, 6.9364462
1: -2.9884815, 3.4774871, -2.5326657, 2.8990409, -5.8875227, 6.0101528
2: -4.5026345, 3.4091501, -3.7398534, 2.8809152, -7.3835497, 7.1490035
3: -5.4607325, 2.7503927, -4.6183352, 2.2743311, -7.7350636, 7.3687277
4: -5.7918782, 3.5346246, -4.5979280, 2.9451058, -8.7369843, 8.1325531
5: -4.8415885, 2.8660069, -3.9415157, 2.3713684, -7.2129569, 6.8075228
6: -5.5861835, 3.0816817, -4.6164842, 2.5191770, -8.1053600, 7.6981659
7: -4.1760111, 4.1995449, -3.5171559, 3.5244100, -7.7004213, 7.7167006
8: -6.1051564, 2.9224575, -5.0834904, 2.3709378, -8.4760942, 8.0059481
9: -4.0349669, 4.0644379, -3.3821173, 3.3687475, -7.4037142, 7.4465551

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 97

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5458557, upper bound: 6.5617153
time: 2.55 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7011305, upper bound: 6.6926596
time: 2.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -3.3403542, 2.3054814, -0.7871694, 0.7614649, -4.1018190, 3.0926509
1: -2.2743425, 2.5697360, -0.7332793, 0.7146363, -2.9889789, 3.3030152
2: -3.2749262, 2.6187005, -0.7288214, 0.9713742, -4.2463002, 3.3475218
3: -3.9970012, 2.0652761, -0.7538663, 0.6586425, -4.6556435, 2.8191423
4: -4.1852818, 2.7175813, -0.9131148, 1.0177121, -5.2029939, 3.6306961
5: -3.5119777, 2.2079182, -0.8726110, 0.8380694, -4.3500471, 3.0805292
6: -3.9842091, 2.4718862, -0.5671178, 1.3561939, -5.3404031, 3.0390038
7: -3.1130657, 3.1289914, -0.8740191, 0.9269050, -4.0399709, 4.0030107
8: -4.4233923, 2.2392547, -0.9697951, 0.8274502, -5.2508426, 3.2090497
9: -2.9995840, 3.0271506, -0.8123494, 0.9296610, -3.9292450, 3.8395000

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7545414, upper bound: 6.8137653
time: 7.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8225903
time: 2.85 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -3.4289427, 2.3639646, -2.7118158, 1.8098432, -5.2387857, 5.0757804
1: -2.3257186, 2.6333265, -1.8506063, 2.1028707, -4.4285893, 4.4839330
2: -3.3630903, 2.6741440, -2.5647974, 2.2254040, -5.5884943, 5.2389412
3: -4.1028318, 2.1140106, -3.2463615, 1.7030371, -5.8058691, 5.3603721
4: -4.2961779, 2.7741041, -3.2477651, 2.3596156, -6.6557932, 6.0218692
5: -3.6064243, 2.2582190, -2.7837048, 1.8283004, -5.4347248, 5.0419235
6: -4.0950108, 2.5144646, -3.3296089, 1.9936430, -6.0886536, 5.8440733
7: -3.1881344, 3.2042284, -2.4828866, 2.5695636, -5.7576981, 5.6871147
8: -4.5424814, 2.2886329, -3.4452233, 1.8256186, -6.3681002, 5.7338562
9: -3.0728631, 3.0999172, -2.4251492, 2.4217777, -5.4946408, 5.5250664

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6577111, upper bound: 6.7071040
time: 2.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8130231
time: 3.14 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.4809000, 1.7183148, -3.4876711, 2.2556217, -4.7365217, 5.2059860
1: -1.7419517, 1.9257169, -2.3425088, 2.6689959, -4.4109478, 4.2682257
2: -2.3865440, 2.0493274, -3.4097962, 2.6590350, -5.0455790, 5.4591236
3: -2.9290504, 1.5728077, -4.2476263, 2.1183224, -5.0473728, 5.8204341
4: -3.0270648, 2.1140773, -4.2339892, 2.7701359, -5.7972007, 6.3480663
5: -2.5746961, 1.7149152, -3.6447158, 2.1936378, -4.7683339, 5.3596311
6: -2.8241107, 2.0015328, -4.2349706, 2.3471055, -5.1712160, 6.2365036
7: -2.3431263, 2.3687315, -3.2071748, 3.2540653, -5.5971918, 5.5759063
8: -3.2126377, 1.7067480, -4.4472165, 2.2292018, -5.4418392, 6.1539645
9: -2.2613215, 2.2860937, -3.1382883, 3.0555551, -5.3168764, 5.4243822

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6117121, upper bound: 6.6161263
time: 3.16 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7484314, upper bound: 6.7282953
time: 3.79 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.7457199, 3.2912567, -3.5300164, 2.2795181, -7.0252380, 6.8212729
1: -3.1335840, 3.6515238, -2.3680983, 2.7002478, -5.8338318, 6.0196218
2: -4.7373276, 3.5718980, -3.4533935, 2.6850398, -7.4223671, 7.0252914
3: -5.7318563, 2.9132266, -4.3022814, 2.1413507, -7.8732071, 7.2155080
4: -6.0864792, 3.6908693, -4.2860470, 2.7982543, -8.8847332, 7.9769163
5: -5.0865865, 2.9999959, -3.6889381, 2.2150345, -7.3016210, 6.6889343
6: -5.8911142, 3.2048693, -4.2933950, 2.3646579, -8.2557716, 7.4982643
7: -4.3768816, 4.3992434, -3.2442248, 3.2908106, -7.6676922, 7.6434679
8: -6.4169745, 3.0850842, -4.5022616, 2.2529869, -8.6699619, 7.5873461
9: -4.2323198, 4.2546630, -3.1751769, 3.0892560, -7.3215761, 7.4298401

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5921513, upper bound: 6.6157299
time: 2.55 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7281945, upper bound: 6.7281945
time: 2.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.23 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5893154, upper bound: 6.5743473
IS_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7896225, upper bound: 6.7855617
IS_A1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5733636, upper bound: 6.5873418
IS_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7069835, upper bound: 6.7351182
IS_A1_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6888792, upper bound: 6.7281810
IS_A1_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7650037, upper bound: 6.8190047
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5902119, upper bound: 6.6349693
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7289493, upper bound: 6.7648816
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5870661, upper bound: 6.6395115
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7300881, upper bound: 6.7789623
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5211878, upper bound: 6.5632971
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6933928, upper bound: 6.7239822
IS_A1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7267007, upper bound: 6.7052129
IS_A1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.8206997, upper bound: 6.8066632
IS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6281823, upper bound: 6.6357824
IS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7289095, upper bound: 6.7592087
IS_A1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7324043, upper bound: 6.7692941
IS_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7457763, upper bound: 6.7843197
IS_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7298808, upper bound: 6.7570433
IS_A1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7436586, upper bound: 6.7722368
IS_A1_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7199241, upper bound: 6.7289159
IS_A1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7174622, upper bound: 6.7051404
IS_A1_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7576588, upper bound: 6.7726793
IS_A1_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7567487, upper bound: 6.7497386
IS_A1_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6795745, upper bound: 6.7141418
IS_A1_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8108016
IS_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6336451, upper bound: 6.6735222
IS_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8015373
IS_A1_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7568891, upper bound: 6.7709936
IS_A1_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7568336, upper bound: 6.7529335
IS_A1_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7700792, upper bound: 6.7873137
IS_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7698538, upper bound: 6.7698538
IS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.8900454, upper bound: 6.8739922
IS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.8643366, upper bound: 6.8697334
IS_A2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.9329495, upper bound: 6.9070298
IS_A2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -7.0107864, upper bound: 6.9734072
IS_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7466009, upper bound: 6.7729710
IS_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7415226, upper bound: 6.7586519
IS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7306932, upper bound: 6.6934277
IS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7260309, upper bound: 6.6933928
IS_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.9042092, upper bound: 6.8424755
IS_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.8877906, upper bound: 6.8420275
IS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -7.0156724, upper bound: 6.9796446
IS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -7.0166256, upper bound: 6.9796446
IS_A2_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7737444, upper bound: 6.7971220
IS_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7710483, upper bound: 6.7852799
IS_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7671955, upper bound: 6.7289493
IS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7608279, upper bound: 6.7289095
IS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.4843250, upper bound: 6.5236738
IS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7566436, upper bound: 6.7860628
IS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5562797, upper bound: 6.6039976
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7073368, upper bound: 6.7494012
IS_A2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5562544, upper bound: 6.5625495
IS_A2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7124061, upper bound: 6.6927644
IS_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5458557, upper bound: 6.5617153
IS_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7011305, upper bound: 6.6926596
IS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7545414, upper bound: 6.8137653
IS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7646079, upper bound: 6.8225903
IS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6577111, upper bound: 6.7071040
IS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7644878, upper bound: 6.8130231
IS_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.6117121, upper bound: 6.6161263
IS_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7484314, upper bound: 6.7282953
IS_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.5921513, upper bound: 6.6157299
IS_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 6, lower bound: -6.7281945, upper bound: 6.7281945

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -1.4513314, 1.1116571, -1.3011838, 1.0290122, -2.4803436, 2.4128408
1: -1.1192150, 1.1620309, -1.0342444, 1.0488709, -2.1680861, 2.1962752
2: -1.3249855, 1.4058700, -1.1753157, 1.3110789, -2.6360645, 2.5811858
3: -1.5940378, 1.0066329, -1.4019890, 0.9244543, -2.5184922, 2.4086218
4: -1.7071179, 1.4778417, -1.5260601, 1.3755991, -3.0827169, 3.0039020
5: -1.5151217, 1.1807904, -1.3610942, 1.1046597, -2.6197815, 2.5418847
6: -1.5105453, 1.5601856, -1.2942221, 1.5110251, -3.0215702, 2.8544078
7: -1.4186064, 1.4709818, -1.2865775, 1.3424007, -2.7610071, 2.7575593
8: -1.7827570, 1.2241277, -1.5941454, 1.1319524, -2.9147096, 2.8182731
9: -1.3673704, 1.4300799, -1.2351745, 1.3141261, -2.6814966, 2.6652546

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5893154, upper bound: 6.5743473
time: 3.23 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.5893154, upper bound: 6.5743473
time: 2.28 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -1.2991980, 1.0289309, -1.4909706, 1.1306081, -2.4298062, 2.5199015
1: -1.0330852, 1.0487301, -1.1444873, 1.1859951, -2.2190804, 2.1932173
2: -1.1753857, 1.3101519, -1.3597590, 1.4279195, -2.6033053, 2.6699109
3: -1.3992611, 0.9244065, -1.6462803, 1.0249015, -2.4241626, 2.5706868
4: -1.5236541, 1.3725073, -1.7579871, 1.5008786, -3.0245328, 3.1304946
5: -1.3611394, 1.1023102, -1.5511426, 1.1996595, -2.5607989, 2.6534529
6: -1.2923901, 1.5061653, -1.5539870, 1.5819613, -2.8743515, 3.0601523
7: -1.2847934, 1.3415102, -1.4532691, 1.5006943, -2.7854877, 2.7947793
8: -1.5926144, 1.1282037, -1.8343141, 1.2322799, -2.8248944, 2.9625177
9: -1.2354445, 1.3135042, -1.3967164, 1.4586569, -2.6941013, 2.7102206

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7896225, upper bound: 6.7855617
time: 3.16 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.7896225, upper bound: 6.7855617
time: 3.23 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -2.8942475, 1.9134451, -0.2882842, 0.3558055, -3.2500529, 2.2017293
1: -1.9517930, 2.2173586, -0.3042573, 0.3640369, -2.3158298, 2.5216160
2: -2.7655163, 2.2967889, -0.2855620, 0.4918183, -3.2573347, 2.5823510
3: -3.4654124, 1.7643008, -0.3150014, 0.2781710, -3.7435834, 2.0793023
4: -3.4936724, 2.2910864, -0.3788152, 0.4079390, -3.9016113, 2.6699016
5: -3.0104809, 1.8707716, -0.4053166, 0.3997480, -3.4102290, 2.2760882
6: -3.2800255, 2.1234808, 0.3243349, 1.2221320, -4.5021572, 1.7991458
7: -2.7053556, 2.7224245, -0.4226231, 0.3896002, -3.0949559, 3.1450477
8: -3.7699840, 1.8518493, -0.3701519, 0.3992378, -4.1692219, 2.2220011
9: -2.5799453, 2.6132569, -0.3572653, 0.3855751, -2.9655204, 2.9705222

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4554973, upper bound: 6.4572015
time: 2.22 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.4462648, upper bound: 6.4569744
time: 2.57 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.6307709, 2.3477173, -0.4902492, 0.5806019, -4.2113729, 2.8379664
1: -2.4291971, 2.7685742, -0.5249176, 0.5303081, -2.9595051, 3.2934918
2: -3.5612109, 2.7747636, -0.5070072, 0.7263057, -4.2875166, 3.2817707
3: -4.4048653, 2.1807127, -0.4775043, 0.4760330, -4.8808985, 2.6582170
4: -4.3916283, 2.8481855, -0.5969779, 0.7474833, -5.1391115, 3.4451635
5: -3.7651315, 2.2813673, -0.6275593, 0.6342238, -4.3993554, 2.9089265
6: -4.3574429, 2.4502311, -0.1114469, 1.2760048, -5.6334476, 2.5616779
7: -3.3838131, 3.3712630, -0.6464683, 0.6366919, -4.0205050, 4.0177312
8: -4.8298388, 2.3061039, -0.6128410, 0.6570655, -5.4869041, 2.9189448
9: -3.2321203, 3.2318654, -0.5779600, 0.6564351, -3.8885555, 3.8098254

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6374088, upper bound: 6.6456921
time: 3.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -6.6171954, upper bound: 6.6442541
time: 3.12 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -1.4984037, 1.1354588, -1.2609806, 1.0071816, -2.5055852, 2.3964396
1: -1.1484270, 1.1931626, -1.0103524, 1.0206424, -2.1690693, 2.2035151
2: -1.3695791, 1.4333899, -1.1352930, 1.2863128, -2.6558919, 2.5686829
3: -1.6524806, 1.0294814, -1.3494436, 0.9040745, -2.5565553, 2.3789248
4: -1.7625823, 1.5106025, -1.4772630, 1.3463422, -3.1089244, 2.9878654
5: -1.5586597, 1.2054155, -1.3221774, 1.0825348, -2.6411943, 2.5275929
6: -1.5755389, 1.5881450, -1.2366548, 1.4904425, -3.0659814, 2.8247998
7: -1.4596220, 1.5070963, -1.2516540, 1.3102357, -2.7698579, 2.7587504
8: -1.8401377, 1.2558317, -1.5450132, 1.0975960, -2.9377337, 2.8008449
9: -1.4045957, 1.4640718, -1.2020448, 1.2841084, -2.6887040, 2.6661167

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=7.834464073181152
rel_dist={6: [-7.106923203393791, 7.106923203247483]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1802.78 seconds
