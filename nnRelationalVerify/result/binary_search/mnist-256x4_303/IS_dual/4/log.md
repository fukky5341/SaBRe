## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 10.8418399842
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673)
1: (-5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646)
2: (-7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834)
3: (-7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134)
4: (-8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527)
5: (-7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887)
6: (-6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578)
7: (-7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368)
8: (-9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864)
9: (-6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577)

## BASE Result
execution time: IAR + LP analysis = 1.27 + 4.19 = 5.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -10.8418977, upper bound: 10.8418970


# Binary Search by BASE starts (time budget: 2694.54 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=13.22586727142334
rel_dist={0: [-10.841896644575556, 10.841898312561103]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=13.22586727142334
rel_dist={0: [-10.841896423745458, 10.841896847270611]}

## Binary Search Result
Binary search time: 18.95 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2675.59 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418855, upper bound: 10.8418498
time: 3.49 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418502, upper bound: 10.8418509
time: 2.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.88
Output dim: 0, lower bound: -10.8418855, upper bound: 10.8418498
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.88
Output dim: 0, lower bound: -10.8418502, upper bound: 10.8418509

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -8.7459450, 4.4799228, -12.9330826, 13.0691080
1: -5.5820465, 5.3603401, -5.7760954, 5.5418692, -11.1239147, 11.1364355
2: -7.2538810, 5.4287362, -7.5248165, 5.6160679, -12.8699474, 12.9535513
3: -7.6901236, 4.7797179, -7.9764252, 4.9407926, -12.6309166, 12.7561417
4: -8.3745241, 6.6817288, -8.6739941, 6.9140596, -15.2885809, 15.3557224
5: -6.9095788, 4.9000840, -7.1571879, 5.0683022, -11.9778805, 12.0572701
6: -6.4428177, 6.5433078, -6.6823978, 6.7787600, -13.2215748, 13.2257061
7: -7.1115313, 6.5268288, -7.3757725, 6.7552648, -13.8667936, 13.9026012
8: -9.0736485, 5.5359268, -9.3987417, 5.7281442, -14.8017912, 14.9346676
9: -6.3155327, 6.4854898, -6.5382051, 6.7157540, -13.0312862, 13.0236950

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417880, upper bound: 10.8418068
time: 3.00 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418165, upper bound: 10.8418212
time: 5.01 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418645, upper bound: 10.8418124
time: 3.38 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418841, upper bound: 10.8418486
time: 2.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7146721, 4.8798447, -8.5726509, 4.3891325, -14.1038036, 13.4524956
1: -6.4488497, 6.1833115, -5.6613750, 5.4347105, -11.8835602, 11.8446865
2: -8.4320307, 6.2204180, -7.3652573, 5.5059786, -13.9380093, 13.5856743
3: -8.9891014, 5.4827590, -7.8071871, 4.8459086, -13.8350105, 13.2899456
4: -9.7494392, 7.7245536, -8.4969816, 6.7768564, -16.5262947, 16.2215347
5: -7.9892755, 5.5848994, -7.0115376, 4.9689403, -12.9582157, 12.5964375
6: -7.4389572, 7.5312815, -6.5408449, 6.6400719, -14.0790291, 14.0721235
7: -8.2275219, 7.4930902, -7.2211084, 6.6209822, -14.8485012, 14.7141991
8: -10.5208874, 6.3528190, -9.2069874, 5.6145005, -16.1353874, 15.5598068
9: -7.3032880, 7.4951725, -6.4073138, 6.5802774, -13.8835621, 13.9024868

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418483
time: 2.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418507, upper bound: 10.8418504
time: 3.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 11.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 11.43
Output dim: 0, lower bound: -10.8418645, upper bound: 10.8418124
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 11.43
Output dim: 0, lower bound: -10.8418841, upper bound: 10.8418486
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 11.43
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418483
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 11.43
Output dim: 0, lower bound: -10.8418507, upper bound: 10.8418504

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.2612820, 4.2272730, -9.0317888, 4.5909619, -12.8522434, 13.2590618
1: -5.4549599, 5.2418756, -5.9846716, 5.7464490, -11.2014084, 11.2265472
2: -7.0780087, 5.3083210, -7.8036337, 5.7975821, -12.8755913, 13.1119547
3: -7.5010548, 4.6752458, -8.2917328, 5.1087627, -12.6098175, 12.9669781
4: -8.1785765, 6.5287199, -9.0197535, 7.1634836, -15.3420601, 15.5484734
5: -6.7478170, 4.7910285, -7.4091954, 5.2150602, -11.9628773, 12.2002239
6: -6.2882767, 6.3912358, -6.9073391, 7.0013695, -13.2896461, 13.2985744
7: -6.9443798, 6.3803139, -7.6415348, 6.9798117, -13.9241915, 14.0218487
8: -8.8603725, 5.4108577, -9.7452393, 5.9135966, -14.7739697, 15.1560974
9: -6.1717105, 6.3360353, -6.7799187, 6.9568548, -13.1285648, 13.1159534

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417518, upper bound: 10.8417517
time: 2.64 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418655, upper bound: 10.8418131
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418640, upper bound: 10.8418123
time: 2.68 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -8.4443331, 4.3287029, -12.7818623, 12.7674971
1: -5.5820465, 5.3603401, -5.5774970, 5.3573160, -10.9393625, 10.9378347
2: -7.2538810, 5.4287362, -7.2505536, 5.4277306, -12.6816092, 12.6792898
3: -7.6901236, 4.7797179, -7.6821594, 4.7775450, -12.4676685, 12.4618778
4: -8.3745241, 6.6817288, -8.3683939, 6.6758642, -15.0503883, 15.0501213
5: -6.9095788, 4.9000840, -6.9050064, 4.8966637, -11.8062420, 11.8050900
6: -6.4428177, 6.5433078, -6.4391675, 6.5410366, -12.9838524, 12.9824753
7: -7.1115313, 6.5268288, -7.1147690, 6.5258894, -13.6374187, 13.6415977
8: -9.0736485, 5.5359268, -9.0669022, 5.5316124, -14.6052599, 14.6028280
9: -6.3155327, 6.4854898, -6.3141775, 6.4829578, -12.7984905, 12.7996664

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417864, upper bound: 10.8418076
time: 4.16 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418147, upper bound: 10.8418194
time: 2.70 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417396, upper bound: 10.8417520
time: 2.94 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418831, upper bound: 10.8418485
time: 3.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418849, upper bound: 10.8418494
time: 2.95 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.3913918, 4.7206936, -6.9052420, 3.5690186, -12.9604101, 11.6259356
1: -6.2376256, 5.9846182, -4.5789948, 4.4114027, -10.6490288, 10.5636129
2: -8.1416101, 6.0201564, -5.8701186, 4.4794159, -12.6210260, 11.8902750
3: -8.6766491, 5.3077612, -6.2050667, 3.9448881, -12.6215372, 11.5128279
4: -9.4221268, 7.4716005, -6.8179293, 5.4762201, -14.8983469, 14.2895298
5: -7.7233911, 5.4003925, -5.6420240, 4.0145016, -11.7378922, 11.0424166
6: -7.1765251, 7.2788348, -5.1838245, 5.3401947, -12.5167198, 12.4626598
7: -7.9484358, 7.2506080, -5.7927351, 5.3697925, -13.3182278, 13.0433426
8: -10.1683159, 6.1432900, -7.3923063, 4.5354104, -14.7037258, 13.5355968
9: -7.0640359, 7.2474756, -5.1801672, 5.3076735, -12.3717098, 12.4276428

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418117, upper bound: 10.8418393
time: 2.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418473
time: 2.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7146721, 4.8798447, -7.9791012, 4.0951805, -13.8098526, 12.8589458
1: -6.4488497, 6.1833115, -5.2715230, 5.0675936, -11.5164433, 11.4548340
2: -8.4320307, 6.2204180, -6.8273439, 5.1371570, -13.5691872, 13.0477619
3: -8.9891014, 5.4827590, -7.2291021, 4.5229588, -13.5120602, 12.7118607
4: -9.7494392, 7.7245536, -7.8917408, 6.3096333, -16.0590725, 15.6162949
5: -7.9892755, 5.5848994, -6.5190420, 4.6305351, -12.6198101, 12.1039410
6: -7.4389572, 7.5312815, -6.0583162, 6.1744084, -13.6133652, 13.5895977
7: -8.2275219, 7.4930902, -6.7067857, 6.1710391, -14.3985615, 14.1998758
8: -10.5208874, 6.3528190, -8.5547352, 5.2289214, -15.7498093, 14.9075546
9: -7.3032880, 7.4951725, -5.9655013, 6.1220951, -13.4253826, 13.4606743

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418421, upper bound: 10.8418130
time: 2.55 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418488, upper bound: 10.8418497
time: 2.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 12.11 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418655, upper bound: 10.8418131
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418640, upper bound: 10.8418123
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418831, upper bound: 10.8418485
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418849, upper bound: 10.8418494
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418117, upper bound: 10.8418393
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418473
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418421, upper bound: 10.8418130
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 12.11
Output dim: 0, lower bound: -10.8418488, upper bound: 10.8418497

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.5793252, 3.4023910, -8.7012024, 4.4296865, -11.0090122, 12.1035938
1: -4.3659754, 4.2105689, -5.7680459, 5.5423059, -9.9082813, 9.9786148
2: -5.5722742, 4.2744532, -7.5062056, 5.5929232, -11.1651974, 11.7806587
3: -5.9047718, 3.7671688, -7.9713287, 4.9294300, -10.8342018, 11.7384977
4: -6.4887624, 5.2175288, -8.6832771, 6.9043040, -13.3930664, 13.9008064
5: -5.3673792, 3.8303609, -7.1371546, 5.0270705, -10.3944492, 10.9675159
6: -4.9206204, 5.0841241, -6.6388869, 6.7431002, -11.6637211, 11.7230110
7: -5.5050707, 5.1241179, -7.3561320, 6.7311339, -12.2362041, 12.4802494
8: -7.0320249, 4.3266668, -9.3840752, 5.6993327, -12.7313576, 13.7107420
9: -4.9347591, 5.0593739, -6.5345774, 6.7031507, -11.6379099, 11.5939512

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417878, upper bound: 10.8417788
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418617, upper bound: 10.8418070
time: 3.17 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418617, upper bound: 10.8418125
time: 2.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.6695352, 3.9352720, -9.0317888, 4.5909619, -12.2604971, 12.9670610
1: -5.0675201, 4.8759274, -5.9846716, 5.7464490, -10.8139687, 10.8605995
2: -6.5432673, 4.9414244, -7.8036337, 5.7975821, -12.3408489, 12.7450581
3: -6.9264612, 4.3534708, -8.2917328, 5.1087627, -12.0352240, 12.6452036
4: -7.5772190, 6.0631118, -9.0197535, 7.1634836, -14.7407026, 15.0828648
5: -6.2569571, 4.4537525, -7.4091954, 5.2150602, -11.4720173, 11.8629475
6: -5.8080854, 5.9281435, -6.9073391, 7.0013695, -12.8094549, 12.8354826
7: -6.4324107, 5.9335518, -7.6415348, 6.9798117, -13.4122219, 13.5750866
8: -8.2100849, 5.0281849, -9.7452393, 5.9135966, -14.1236820, 14.7734241
9: -5.7313566, 5.8799062, -6.7799187, 6.9568548, -12.6882114, 12.6598244

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417504, upper bound: 10.8417522
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418615, upper bound: 10.8418077
time: 2.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418615, upper bound: 10.8418131
time: 2.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7842145, 3.5019553, -8.1179886, 4.1686935, -10.9529076, 11.6199436
1: -4.4997196, 4.3365183, -5.3634996, 5.1553698, -9.6550894, 9.7000179
2: -5.7580748, 4.4015784, -6.9556284, 5.2253942, -10.9834690, 11.3572063
3: -6.0936804, 3.8780322, -7.3646774, 4.6001081, -10.6937885, 11.2427101
4: -6.6953669, 5.3803387, -8.0360155, 6.4191103, -13.1144772, 13.4163542
5: -5.5390244, 3.9451072, -6.6350331, 4.7106371, -10.2496614, 10.5801401
6: -5.0846615, 5.2438340, -6.1738381, 6.2852468, -11.3699083, 11.4176722
7: -5.6818085, 5.2764645, -6.8328471, 6.2794456, -11.9612541, 12.1093121
8: -7.2579589, 4.4572582, -8.7086306, 5.3200130, -12.5779724, 13.1658888
9: -5.0875573, 5.2140598, -6.0716624, 6.2314310, -11.3189888, 11.2857227

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418139, upper bound: 10.8418187
time: 2.82 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418841, upper bound: 10.8418465
time: 3.00 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418841, upper bound: 10.8418489
time: 2.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.8623919, 4.0305882, -8.4443331, 4.3287029, -12.1910944, 12.4749212
1: -5.1942968, 4.9949150, -5.5774970, 5.3573160, -10.5516129, 10.5724115
2: -6.7187901, 5.0618858, -7.2505536, 5.4277306, -12.1465206, 12.3124371
3: -7.1151147, 4.4582224, -7.6821594, 4.7775450, -11.8926601, 12.1403818
4: -7.7725391, 6.2167931, -8.3683939, 6.6758642, -14.4484034, 14.5851860
5: -6.4192629, 4.5632172, -6.9050064, 4.8966637, -11.3159266, 11.4682236
6: -5.9627509, 6.0802150, -6.4391675, 6.5410366, -12.5037880, 12.5193806
7: -6.5997624, 6.0794239, -7.1147690, 6.5258894, -13.1256523, 13.1941929
8: -8.4242134, 5.1527052, -9.0669022, 5.5316124, -13.9558249, 14.2196074
9: -5.8758101, 6.0295091, -6.3141775, 6.4829578, -12.3587685, 12.3436871

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417871, upper bound: 10.8418067
time: 3.03 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418831, upper bound: 10.8418458
time: 3.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418831, upper bound: 10.8418483
time: 3.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1344662, 4.5917859, -6.9052420, 3.5690186, -12.7034845, 11.4970284
1: -6.0705981, 5.8291759, -4.5789948, 4.4114027, -10.4820004, 10.4081707
2: -7.9111710, 5.8603606, -5.8701186, 4.4794159, -12.3905869, 11.7304792
3: -8.4291124, 5.1697078, -6.2050667, 3.9448881, -12.3740005, 11.3747749
4: -9.1665335, 7.2705598, -6.8179293, 5.4762201, -14.6427536, 14.0884895
5: -7.5107622, 5.2530956, -5.6420240, 4.0145016, -11.5252638, 10.8951197
6: -6.9697409, 7.0772281, -5.1838245, 5.3401947, -12.3099356, 12.2610531
7: -7.7274656, 7.0586877, -5.7927351, 5.3697925, -13.0972576, 12.8514233
8: -9.8883305, 5.9771619, -7.3923063, 4.5354104, -14.4237404, 13.3694687
9: -6.8751688, 7.0510020, -5.1801672, 5.3076735, -12.1828423, 12.2311687

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418402
time: 3.05 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418146
time: 3.18 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.5623264, 4.8035049, -8.2824879, 4.2180119, -13.7803383, 13.0859928
1: -6.3489566, 6.0904441, -5.4939914, 5.2840452, -11.6330013, 11.5844355
2: -8.2941952, 6.1250057, -7.1275663, 5.3313475, -13.6255426, 13.2525721
3: -8.8407612, 5.4005175, -7.5662279, 4.7014561, -13.5422173, 12.9667454
4: -9.5965233, 7.6041584, -8.2583942, 6.5758815, -16.1724052, 15.8625526
5: -7.8620234, 5.4978728, -6.7905531, 4.7874417, -12.6494656, 12.2884254
6: -7.3165402, 7.4109840, -6.2977867, 6.4139957, -13.7305355, 13.7087708
7: -8.0954561, 7.3786063, -6.9895868, 6.4135561, -14.5090122, 14.3681927
8: -10.3532171, 6.2541752, -8.9258423, 5.4273748, -15.7805920, 15.1800175
9: -7.1901665, 7.3776336, -6.2221403, 6.3800964, -13.5702629, 13.5997734

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418119
time: 2.67 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418124
time: 2.58 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.7146721, 4.8798447, -7.6718569, 3.9429567, -13.6576290, 12.5517015
1: -6.4488497, 6.1833115, -5.0709624, 4.8794298, -11.3282795, 11.2542744
2: -8.4320307, 6.2204180, -6.5498638, 4.9463010, -13.3783321, 12.7702818
3: -8.9891014, 5.4827590, -6.9313526, 4.3567467, -13.3458481, 12.4141121
4: -9.7494392, 7.7245536, -7.5830636, 6.0670605, -15.8164997, 15.3076172
5: -7.9892755, 5.5848994, -6.2624788, 4.4555721, -12.4448471, 11.8473778
6: -7.4389572, 7.5312815, -5.8114376, 5.9331975, -13.3721542, 13.3427191
7: -8.2275219, 7.4930902, -6.4420705, 5.9397602, -14.1672821, 13.9351606
8: -10.5208874, 6.3528190, -8.2164974, 5.0308194, -15.5517063, 14.5693169
9: -7.3032880, 7.4951725, -5.7375031, 5.8855348, -13.1888227, 13.2326756

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418478
time: 2.32 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418489
time: 16.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418617, upper bound: 10.8418070
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418617, upper bound: 10.8418125
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418615, upper bound: 10.8418077
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418615, upper bound: 10.8418131
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418841, upper bound: 10.8418465
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418841, upper bound: 10.8418489
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418831, upper bound: 10.8418458
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418831, upper bound: 10.8418483
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418402
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418146
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418119
IS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418124
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418478
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.27
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418489

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.5793252, 3.4023910, -7.2726078, 3.7351594, -10.3144846, 10.6749992
1: -4.3659754, 4.2105689, -4.8354149, 4.6592951, -9.0252705, 9.0459843
2: -5.5722742, 4.2744532, -6.2216101, 4.7130747, -10.2853489, 10.4960632
3: -5.9047718, 3.7671688, -6.5886726, 4.1544385, -10.0592098, 10.3558416
4: -6.4887624, 5.2175288, -7.2312374, 5.7844186, -12.2731810, 12.4487667
5: -5.3673792, 3.8303609, -5.9598875, 4.2146621, -9.5820408, 9.7902489
6: -4.9206204, 5.0841241, -5.4810200, 5.6310587, -10.5516796, 10.5651436
7: -5.5050707, 5.1241179, -6.1286831, 5.6578279, -11.1628990, 11.2528009
8: -7.0320249, 4.3266668, -7.8219428, 4.7759857, -11.8080101, 12.1486092
9: -4.9347591, 5.0593739, -5.4758124, 5.6077337, -10.5424929, 10.5351868

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418405, upper bound: 10.8417764
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8417772
time: 2.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.5793252, 3.4023910, -8.4345560, 4.2981105, -10.8774357, 11.8369465
1: -4.3659754, 4.2105689, -5.5933251, 5.3776903, -9.7436657, 9.8038940
2: -5.5722742, 4.2744532, -7.2659316, 5.4274588, -10.9997330, 11.5403843
3: -5.9047718, 3.7671688, -7.7128668, 4.7845964, -10.6893682, 11.4800358
4: -6.4887624, 5.2175288, -8.4118996, 6.6952081, -13.1839705, 13.6294289
5: -5.3673792, 3.8303609, -6.9170418, 4.8754215, -10.2428007, 10.7474022
6: -4.9206204, 5.0841241, -6.4225998, 6.5346985, -11.4553185, 11.5067234
7: -5.5050707, 5.1241179, -7.1249471, 6.5301456, -12.0352163, 12.2490654
8: -7.0320249, 4.3266668, -9.0925350, 5.5265088, -12.5585337, 13.4192019
9: -4.9347591, 5.0593739, -6.3362074, 6.4981527, -11.4329119, 11.3955812

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417859, upper bound: 10.8417779
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418443, upper bound: 10.8418104
time: 2.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418339, upper bound: 10.8417859
time: 2.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.6695352, 3.9352720, -7.2726078, 3.7351594, -11.4046946, 11.2078800
1: -5.0675201, 4.8759274, -4.8354149, 4.6592951, -9.7268152, 9.7113419
2: -6.5432673, 4.9414244, -6.2216101, 4.7130747, -11.2563419, 11.1630344
3: -6.9264612, 4.3534708, -6.5886726, 4.1544385, -11.0809002, 10.9421434
4: -7.5772190, 6.0631118, -7.2312374, 5.7844186, -13.3616371, 13.2943497
5: -6.2569571, 4.4537525, -5.9598875, 4.2146621, -10.4716187, 10.4136400
6: -5.8080854, 5.9281435, -5.4810200, 5.6310587, -11.4391441, 11.4091635
7: -6.4324107, 5.9335518, -6.1286831, 5.6578279, -12.0902386, 12.0622349
8: -8.2100849, 5.0281849, -7.8219428, 4.7759857, -12.9860706, 12.8501282
9: -5.7313566, 5.8799062, -5.4758124, 5.6077337, -11.3390903, 11.3557186

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 113

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418616, upper bound: 10.8418079
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418598, upper bound: 10.8418086
time: 3.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.6695352, 3.9352720, -8.4345560, 4.2981105, -11.9676456, 12.3698282
1: -5.0675201, 4.8759274, -5.5933251, 5.3776903, -10.4452105, 10.4692526
2: -6.5432673, 4.9414244, -7.2659316, 5.4274588, -11.9707260, 12.2073555
3: -6.9264612, 4.3534708, -7.7128668, 4.7845964, -11.7110577, 12.0663376
4: -7.5772190, 6.0631118, -8.4118996, 6.6952081, -14.2724266, 14.4750118
5: -6.2569571, 4.4537525, -6.9170418, 4.8754215, -11.1323786, 11.3707943
6: -5.8080854, 5.9281435, -6.4225998, 6.5346985, -12.3427839, 12.3507433
7: -6.4324107, 5.9335518, -7.1249471, 6.5301456, -12.9625568, 13.0584984
8: -8.2100849, 5.0281849, -9.0925350, 5.5265088, -13.7365932, 14.1207199
9: -5.7313566, 5.8799062, -6.3362074, 6.4981527, -12.2295094, 12.2161140

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418411, upper bound: 10.8417854
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418353, upper bound: 10.8417848
time: 2.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7842145, 3.5019553, -6.7623572, 3.5050676, -10.2892818, 10.2643127
1: -4.4997196, 4.3365183, -4.4866800, 4.3248391, -8.8245583, 8.8231983
2: -5.7580748, 4.4015784, -5.7434225, 4.3932295, -10.1513042, 10.1450005
3: -6.0936804, 3.8780322, -6.0744686, 3.8689585, -9.9626389, 9.9525013
4: -6.6953669, 5.3803387, -6.6757946, 5.3634815, -12.0588484, 12.0561333
5: -5.5390244, 3.9451072, -5.5238981, 3.9356031, -9.4746275, 9.4690056
6: -5.0846615, 5.2438340, -5.0713344, 5.2310257, -10.3156872, 10.3151684
7: -5.6818085, 5.2764645, -5.6754498, 5.2665067, -10.9483147, 10.9519138
8: -7.2579589, 4.4572582, -7.2366996, 4.4454107, -11.7033691, 11.6939583
9: -5.0875573, 5.2140598, -5.0762796, 5.2023282, -10.2898855, 10.2903395

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418142, upper bound: 10.8418186
time: 2.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417523, upper bound: 10.8417739
time: 2.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418636, upper bound: 10.8418145
time: 3.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418565, upper bound: 10.8418144
time: 2.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7842145, 3.5019553, -7.8462601, 4.0339379, -10.8181524, 11.3482151
1: -4.4997196, 4.3365183, -5.1852961, 4.9872808, -9.4870005, 9.5218143
2: -5.7580748, 4.4015784, -6.7095795, 5.0565295, -10.8146038, 11.1111584
3: -6.0936804, 3.8780322, -7.1005392, 4.4521403, -10.5458202, 10.9785709
4: -6.6953669, 5.3803387, -7.7596312, 6.2049861, -12.9003525, 13.1399698
5: -5.5390244, 3.9451072, -6.4089384, 4.5558977, -10.0949221, 10.3540459
6: -5.0846615, 5.2438340, -5.9536190, 6.0721517, -11.1568127, 11.1974525
7: -5.6818085, 5.2764645, -6.5970979, 6.0738144, -11.7556229, 11.8735619
8: -7.2579589, 4.4572582, -8.4094000, 5.1443090, -12.4022675, 12.8666582
9: -5.0875573, 5.2140598, -5.8689585, 6.0215111, -11.1090679, 11.0830183

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418142, upper bound: 10.8418198
time: 2.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418634, upper bound: 10.8418445
time: 3.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418565, upper bound: 10.8418185
time: 2.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.8623919, 4.0305882, -6.7623572, 3.5050676, -11.3674593, 10.7929459
1: -5.1942968, 4.9949150, -4.4866800, 4.3248391, -9.5191364, 9.4815950
2: -6.7187901, 5.0618858, -5.7434225, 4.3932295, -11.1120195, 10.8053083
3: -7.1151147, 4.4582224, -6.0744686, 3.8689585, -10.9840736, 10.5326910
4: -7.7725391, 6.2167931, -6.6757946, 5.3634815, -13.1360207, 12.8925877
5: -6.4192629, 4.5632172, -5.5238981, 3.9356031, -10.3548660, 10.0871153
6: -5.9627509, 6.0802150, -5.0713344, 5.2310257, -11.1937771, 11.1515493
7: -6.5997624, 6.0794239, -5.6754498, 5.2665067, -11.8662691, 11.7548733
8: -8.4242134, 5.1527052, -7.2366996, 4.4454107, -12.8696241, 12.3894043
9: -5.8758101, 6.0295091, -5.0762796, 5.2023282, -11.0781384, 11.1057892

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417519, upper bound: 10.8417708
time: 3.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418635, upper bound: 10.8418142
time: 2.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418574, upper bound: 10.8418140
time: 2.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.8623919, 4.0305882, -7.8462601, 4.0339379, -11.8963299, 11.8768482
1: -5.1942968, 4.9949150, -5.1852961, 4.9872808, -10.1815777, 10.1802111
2: -6.7187901, 5.0618858, -6.7095795, 5.0565295, -11.7753201, 11.7714653
3: -7.1151147, 4.4582224, -7.1005392, 4.4521403, -11.5672550, 11.5587616
4: -7.7725391, 6.2167931, -7.7596312, 6.2049861, -13.9775257, 13.9764242
5: -6.4192629, 4.5632172, -6.4089384, 4.5558977, -10.9751606, 10.9721556
6: -5.9627509, 6.0802150, -5.9536190, 6.0721517, -12.0349026, 12.0338345
7: -6.5997624, 6.0794239, -6.5970979, 6.0738144, -12.6735764, 12.6765213
8: -8.4242134, 5.1527052, -8.4094000, 5.1443090, -13.5685225, 13.5621052
9: -5.8758101, 6.0295091, -5.8689585, 6.0215111, -11.8973217, 11.8984680

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417519, upper bound: 10.8418066
time: 2.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418139, upper bound: 10.8418207
time: 3.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417706, upper bound: 10.8418014
time: 3.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418635, upper bound: 10.8418179
time: 3.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418574, upper bound: 10.8418170
time: 3.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1344662, 4.5917859, -6.3861103, 3.3194168, -12.4538832, 10.9778957
1: -6.0705981, 5.8291759, -4.2385592, 4.0891347, -10.1597328, 10.0677357
2: -7.9111710, 5.8603606, -5.3977571, 4.1571684, -12.0683393, 11.2581177
3: -8.4291124, 5.1697078, -5.7242117, 3.6626618, -12.0917740, 10.8939190
4: -9.1665335, 7.2705598, -6.2885284, 5.0627332, -14.2292671, 13.5590878
5: -7.5107622, 5.2530956, -5.2073312, 3.7247612, -11.2355232, 10.4604263
6: -6.9697409, 7.0772281, -4.7669063, 4.9359012, -11.9056416, 11.8441343
7: -7.7274656, 7.0586877, -5.3428569, 4.9821963, -12.7096615, 12.4015446
8: -9.8883305, 5.9771619, -6.8181515, 4.2039261, -14.0922565, 12.7953129
9: -6.8751688, 7.0510020, -4.7907162, 4.9140825, -11.7892513, 11.8417187

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418027
time: 2.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418389
time: 3.00 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -7.9389896, 4.0078292, -7.6718569, 3.9429567, -11.8819466, 11.6796856
1: -5.2842102, 5.0867510, -5.0709624, 4.8794298, -10.1636400, 10.1577129
2: -6.8300972, 5.1184130, -6.5498638, 4.9463010, -11.7763977, 11.6682768
3: -7.2633328, 4.5182614, -6.9313526, 4.3567467, -11.6200790, 11.4496136
4: -7.9412947, 6.3284316, -7.5830636, 6.0670605, -14.0083551, 13.9114952
5: -6.5214534, 4.5743227, -6.2624788, 4.4555721, -10.9770260, 10.8368015
6: -6.0010281, 6.1445255, -5.8114376, 5.9331975, -11.9342251, 11.9559631
7: -6.6886630, 6.1578865, -6.4420705, 5.9397602, -12.6284237, 12.5999565
8: -8.5742321, 5.2025962, -8.2164974, 5.0308194, -13.6050510, 13.4190941
9: -5.9821663, 6.1277628, -5.7375031, 5.8855348, -11.8677006, 11.8652658

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 113

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418183, upper bound: 10.8418284
time: 2.51 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418181, upper bound: 10.8418217
time: 2.18 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1093225, 4.5797076, -7.6718569, 3.9429567, -13.0522795, 12.2515640
1: -6.0533729, 5.8115520, -5.0709624, 4.8794298, -10.9328022, 10.8825150
2: -7.8877387, 5.8447852, -6.5498638, 4.9463010, -12.8340397, 12.3946495
3: -8.4040852, 5.1549087, -6.9313526, 4.3567467, -12.7608318, 12.0862617
4: -9.1370716, 7.2508297, -7.5830636, 6.0670605, -15.2041321, 14.8338928
5: -7.4905796, 5.2391715, -6.2624788, 4.4555721, -11.9461517, 11.5016499
6: -6.9476757, 7.0581341, -5.8114376, 5.9331975, -12.8808727, 12.8695717
7: -7.7037601, 7.0385532, -6.4420705, 5.9397602, -13.6435204, 13.4806232
8: -9.8604088, 5.9603915, -8.2164974, 5.0308194, -14.8912277, 14.1768894
9: -6.8548994, 7.0309110, -5.7375031, 5.8855348, -12.7404346, 12.7684135

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418421
time: 2.05 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418138, upper bound: 10.8418176
time: 2.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.51 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418405, upper bound: 10.8417764
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8417772
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418443, upper bound: 10.8418104
IS_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418339, upper bound: 10.8417859
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418616, upper bound: 10.8418079
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418598, upper bound: 10.8418086
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418411, upper bound: 10.8417854
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418353, upper bound: 10.8417848
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418636, upper bound: 10.8418145
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418565, upper bound: 10.8418144
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418634, upper bound: 10.8418445
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418565, upper bound: 10.8418185
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418635, upper bound: 10.8418142
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418574, upper bound: 10.8418140
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418635, upper bound: 10.8418179
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418574, upper bound: 10.8418170
IS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418027
IS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418389
IS_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418183, upper bound: 10.8418284
IS_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418181, upper bound: 10.8418217
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418421
IS_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 11.51
Output dim: 0, lower bound: -10.8418138, upper bound: 10.8418176

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.0848489, 3.1653268, -7.2726078, 3.7351594, -9.8200083, 10.4379349
1: -4.0379601, 3.8994198, -4.8354149, 4.6592951, -8.6972551, 8.7348347
2: -5.1177402, 3.9658325, -6.2216101, 4.7130747, -9.8308144, 10.1874428
3: -5.4393864, 3.4960358, -6.5886726, 4.1544385, -9.5938244, 10.0847082
4: -5.9761662, 4.8193641, -7.2312374, 5.7844186, -11.7605848, 12.0506020
5: -4.9487019, 3.5565515, -5.9598875, 4.2146621, -9.1633644, 9.5164394
6: -4.5247154, 4.6983118, -5.4810200, 5.6310587, -10.1557741, 10.1793318
7: -5.0715775, 4.7515173, -6.1286831, 5.6578279, -10.7294054, 10.8802004
8: -6.4787698, 4.0105824, -7.8219428, 4.7759857, -11.2547550, 11.8325253
9: -4.5582142, 4.6798878, -5.4758124, 5.6077337, -10.1659479, 10.1557007

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418394, upper bound: 10.8417760
time: 2.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418403, upper bound: 10.8417764
time: 2.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -6.5793252, 3.4023910, -8.0078697, 4.0886984, -10.6680241, 11.4102612
1: -4.3659754, 4.2105689, -5.3177691, 5.1172500, -9.4832249, 9.5283375
2: -5.5722742, 4.2744532, -6.8858118, 5.1654625, -10.7377367, 11.1602650
3: -5.9047718, 3.7671688, -7.3057160, 4.5547137, -10.4594860, 11.0728846
4: -6.4887624, 5.2175288, -7.9849043, 6.3647799, -12.8535423, 13.2024326
5: -5.3673792, 3.8303609, -6.5696483, 4.6303082, -9.9976873, 10.4000092
6: -4.9206204, 5.0841241, -6.0746264, 6.2030196, -11.1236401, 11.1587505
7: -5.5050707, 5.1241179, -6.7605114, 6.2108669, -11.7159376, 11.8846292
8: -7.0320249, 4.3266668, -8.6317472, 5.2507286, -12.2827530, 12.9584141
9: -4.9347591, 5.0593739, -6.0243030, 6.1739163, -11.1086750, 11.0836773

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417285, upper bound: 10.8417352
time: 3.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418402, upper bound: 10.8417855
time: 2.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418402, upper bound: 10.8417857
time: 2.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.3442717, 3.2942269, -7.2223668, 3.7099302, -10.0542021, 10.5165939
1: -4.1846685, 4.0373044, -4.8020277, 4.6276188, -8.8122873, 8.8393326
2: -5.3228021, 4.1123533, -6.1753621, 4.6814609, -10.0042629, 10.2877159
3: -5.6319599, 3.6233053, -6.5390077, 4.1266809, -9.7586403, 10.1623135
4: -6.1931920, 4.9959774, -7.1789513, 5.7441597, -11.9373512, 12.1749287
5: -5.1330690, 3.7154403, -5.9171190, 4.1864624, -9.3195314, 9.6325588
6: -4.7455926, 4.8913097, -5.4406829, 5.5917664, -10.3373585, 10.3319931
7: -5.2654963, 4.9241877, -6.0840302, 5.6191626, -10.8846588, 11.0082178
8: -6.7236271, 4.1739287, -7.7657623, 4.7434893, -11.4671164, 11.9396915
9: -4.7178574, 4.8458161, -5.4372973, 5.5682335, -10.2860909, 10.2831135

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8417769
time: 3.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8417762
time: 3.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.1800427, 3.6865771, -7.2373075, 3.7173154, -10.8973579, 10.9238844
1: -4.7431383, 4.5692110, -4.8119259, 4.6370115, -9.3801498, 9.3811369
2: -6.0928125, 4.6328001, -6.1890297, 4.6907907, -10.7836037, 10.8218298
3: -6.4435015, 4.0841489, -6.5537372, 4.1349106, -10.5784121, 10.6378860
4: -7.0724459, 5.6712856, -7.1944647, 5.7560921, -12.8285379, 12.8657503
5: -5.8405137, 4.1771998, -5.9298086, 4.1948023, -10.0353165, 10.1070080
6: -5.4146619, 5.5422325, -5.4526148, 5.6033535, -11.0180149, 10.9948473
7: -5.9980459, 5.5572853, -6.0971651, 5.6306052, -11.6286507, 11.6544504
8: -7.6627488, 4.7112651, -7.7824011, 4.7531137, -12.4158630, 12.4936657
9: -5.3576274, 5.4954376, -5.4486928, 5.5799198, -10.9375477, 10.9441299

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418387, upper bound: 10.8417761
time: 5.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8417773
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1883240, 3.6979899, -8.4345560, 4.2981105, -11.4864349, 12.1325455
1: -4.7553124, 4.5815864, -5.5933251, 5.3776903, -10.1330032, 10.1749115
2: -6.1106911, 4.6438990, -7.2659316, 5.4274588, -11.5381498, 11.9098301
3: -6.4635963, 4.0940881, -7.7128668, 4.7845964, -11.2481928, 11.8069553
4: -7.0957880, 5.6857142, -8.4118996, 6.6952081, -13.7909966, 14.0976143
5: -5.8595943, 4.1779108, -6.9170418, 4.8754215, -10.7350159, 11.0949526
6: -5.4182863, 5.5514054, -6.4225998, 6.5346985, -11.9529848, 11.9740047
7: -6.0193796, 5.5723596, -7.1249471, 6.5301456, -12.5495253, 12.6973066
8: -7.6847401, 4.7170820, -9.0925350, 5.5265088, -13.2112484, 13.8096170
9: -5.3766012, 5.5120101, -6.3362074, 6.4981527, -11.8747540, 11.8482170

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417261, upper bound: 10.8417228
time: 3.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8417858
time: 3.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8417859
time: 3.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.2674336, 3.2533917, -6.7623572, 3.5050676, -9.7725010, 10.0157490
1: -4.1599512, 4.0148692, -4.4866800, 4.3248391, -8.4847908, 8.5015488
2: -5.2866993, 4.0801983, -5.7434225, 4.3932295, -9.6799288, 9.8236208
3: -5.6131477, 3.5965378, -6.0744686, 3.8689585, -9.4821062, 9.6710062
4: -6.1666198, 4.9675746, -6.6757946, 5.3634815, -11.5301018, 11.6433697
5: -5.1050239, 3.6571298, -5.5238981, 3.9356031, -9.0406265, 9.1810284
6: -4.6700850, 4.8411198, -5.0713344, 5.2310257, -9.9011106, 9.9124546
7: -5.2327018, 4.8899021, -5.6754498, 5.2665067, -10.4992085, 10.5653515
8: -6.6847658, 4.1272063, -7.2366996, 4.4454107, -11.1301765, 11.3639059
9: -4.6985455, 4.8210621, -5.0762796, 5.2023282, -9.9008732, 9.8973417

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418367, upper bound: 10.8417914
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418358, upper bound: 10.8417865
time: 3.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.36 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418394, upper bound: 10.8417760
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418403, upper bound: 10.8417764
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418402, upper bound: 10.8417855
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418402, upper bound: 10.8417857
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8417769
IS_A1_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8417762
IS_A1_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418387, upper bound: 10.8417761
IS_A1_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8417773
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8417858
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8417859
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418367, upper bound: 10.8417914
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.36
Output dim: 0, lower bound: -10.8418358, upper bound: 10.8417865
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418565, upper bound: 10.8418144
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418634, upper bound: 10.8418445
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418565, upper bound: 10.8418185
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418635, upper bound: 10.8418142
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418574, upper bound: 10.8418140
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418635, upper bound: 10.8418179
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418574, upper bound: 10.8418170
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.36
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418421
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418702, upper bound: 10.8418508
time: 2.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418505
time: 2.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.01
Output dim: 0, lower bound: -10.8418702, upper bound: 10.8418508
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.01
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418505

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -8.7459450, 4.4799228, -12.9330826, 13.0691080
1: -5.5820465, 5.3603401, -5.7760954, 5.5418692, -11.1239147, 11.1364355
2: -7.2538810, 5.4287362, -7.5248165, 5.6160679, -12.8699474, 12.9535513
3: -7.6901236, 4.7797179, -7.9764252, 4.9407926, -12.6309166, 12.7561417
4: -8.3745241, 6.6817288, -8.6739941, 6.9140596, -15.2885809, 15.3557224
5: -6.9095788, 4.9000840, -7.1571879, 5.0683022, -11.9778805, 12.0572701
6: -6.4428177, 6.5433078, -6.6823978, 6.7787600, -13.2215748, 13.2257061
7: -7.1115313, 6.5268288, -7.3757725, 6.7552648, -13.8667936, 13.9026012
8: -9.0736485, 5.5359268, -9.3987417, 5.7281442, -14.8017912, 14.9346676
9: -6.3155327, 6.4854898, -6.5382051, 6.7157540, -13.0312862, 13.0236950

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418509
time: 2.15 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418492
time: 3.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7146721, 4.8798447, -8.3293467, 4.2609072, -13.9755793, 13.2091913
1: -6.4488497, 6.1833115, -5.5005522, 5.2841473, -11.7329969, 11.6838636
2: -8.4320307, 6.2204180, -7.1408758, 5.3513956, -13.7834263, 13.3612938
3: -8.9891014, 5.4827590, -7.5695782, 4.7124772, -13.7015781, 13.0523376
4: -9.7494392, 7.7245536, -8.2486372, 6.5840526, -16.3334923, 15.9731903
5: -7.9892755, 5.5848994, -6.8067393, 4.8290186, -12.8182945, 12.3916388
6: -7.4389572, 7.5312815, -6.3417721, 6.4451447, -13.8841019, 13.8730526
7: -8.2275219, 7.4930902, -7.0039182, 6.4320908, -14.6596107, 14.4970083
8: -10.5208874, 6.3528190, -8.9375420, 5.4546876, -15.9755745, 15.2903614
9: -7.3032880, 7.4951725, -6.2235589, 6.3898439, -13.6931314, 13.7187309

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418293, upper bound: 10.8418122
time: 2.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418492
time: 2.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418509
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418492
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 16.37
Output dim: 0, lower bound: -10.8418293, upper bound: 10.8418122
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.37
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418492

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -8.4531593, 4.3231654, -12.7763233, 12.7763243
1: -5.5820465, 5.3603401, -5.5820465, 5.3603401, -10.9423857, 10.9423857
2: -7.2538810, 5.4287362, -7.2538810, 5.4287362, -12.6826153, 12.6826153
3: -7.6901236, 4.7797179, -7.6901236, 4.7797179, -12.4698410, 12.4698410
4: -8.3745241, 6.6817288, -8.3745241, 6.6817288, -15.0562515, 15.0562506
5: -6.9095788, 4.9000840, -6.9095788, 4.9000840, -11.8096619, 11.8096619
6: -6.4428177, 6.5433078, -6.4428177, 6.5433078, -12.9861259, 12.9861259
7: -7.1115313, 6.5268288, -7.1115313, 6.5268288, -13.6383591, 13.6383591
8: -9.0736485, 5.5359268, -9.0736485, 5.5359268, -14.6095715, 14.6095734
9: -6.3155327, 6.4854898, -6.3155327, 6.4854898, -12.8010225, 12.8010225

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417768, upper bound: 10.8417791
time: 3.71 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417544, upper bound: 10.8417628
time: 3.68 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418486, upper bound: 10.8418227
time: 3.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418439, upper bound: 10.8418219
time: 2.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -9.7146721, 4.8798447, -13.3330040, 14.0378361
1: -5.5820465, 5.3603401, -6.4488497, 6.1833115, -11.7653580, 11.8091898
2: -7.2538810, 5.4287362, -8.4320307, 6.2204180, -13.4742966, 13.8607674
3: -7.6901236, 4.7797179, -8.9891014, 5.4827590, -13.1728821, 13.7688198
4: -8.3745241, 6.6817288, -9.7494392, 7.7245536, -16.0990753, 16.4311676
5: -6.9095788, 4.9000840, -7.9892755, 5.5848994, -12.4944782, 12.8893595
6: -6.4428177, 6.5433078, -7.4389572, 7.5312815, -13.9740982, 13.9822655
7: -7.1115313, 6.5268288, -8.2275219, 7.4930902, -14.6046219, 14.7543497
8: -9.0736485, 5.5359268, -10.5208874, 6.3528190, -15.4264679, 16.0568142
9: -6.3155327, 6.4854898, -7.3032880, 7.4951725, -13.8107052, 13.7887783

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 70

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417768, upper bound: 10.8417797
time: 3.75 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418368, upper bound: 10.8418344
time: 3.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418488
time: 3.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.6746130, 4.8597627, -8.0288315, 4.1115885, -13.7862015, 12.8885937
1: -6.4227462, 6.1590152, -5.3037448, 5.1002073, -11.5229530, 11.4627600
2: -8.3960371, 6.1954660, -6.8692131, 5.1641765, -13.5602131, 13.0646791
3: -8.9504013, 5.4612012, -7.2769246, 4.5498085, -13.5002098, 12.7381258
4: -9.7094669, 7.6931381, -7.9460616, 6.3469553, -16.0564232, 15.6392002
5: -7.9560471, 5.5619626, -6.5558634, 4.6576014, -12.6136484, 12.1178265
6: -7.4066987, 7.4998121, -6.0997963, 6.2089782, -13.6156769, 13.5996084
7: -8.1930113, 7.4631186, -6.7443414, 6.2058315, -14.3988428, 14.2074604
8: -10.4771242, 6.3269091, -8.6065540, 5.2607832, -15.7379074, 14.9334631
9: -7.2737603, 7.4644775, -6.0006413, 6.1578093, -13.4315701, 13.4651184

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418121, upper bound: 10.8418297
time: 3.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418126, upper bound: 10.8418489
time: 4.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.95 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.95
Output dim: 0, lower bound: -10.8418486, upper bound: 10.8418227
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.95
Output dim: 0, lower bound: -10.8418439, upper bound: 10.8418219
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 10.95
Output dim: 0, lower bound: -10.8418368, upper bound: 10.8418344
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.95
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418488
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 10.95
Output dim: 0, lower bound: -10.8418121, upper bound: 10.8418297
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.95
Output dim: 0, lower bound: -10.8418126, upper bound: 10.8418489

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.7937489, 4.0099812, -8.3106747, 4.2551985, -12.0489454, 12.3206539
1: -5.1377392, 4.9396811, -5.4859600, 5.2694006, -10.4071388, 10.4256392
2: -6.6431761, 5.0149889, -7.1217260, 5.3390613, -11.9822369, 12.1367149
3: -7.0266523, 4.4149866, -7.5466061, 4.7008138, -11.7274647, 11.9615927
4: -7.6771269, 6.1471119, -8.2236986, 6.5660448, -14.2431698, 14.3708105
5: -6.3495426, 4.5325346, -6.7885251, 4.8205771, -11.1701202, 11.3210602
6: -5.9140644, 6.0239234, -6.3284483, 6.4307137, -12.3447781, 12.3523693
7: -6.5300941, 6.0211878, -6.9855032, 6.4174318, -12.9475260, 13.0066910
8: -8.3281164, 5.1081285, -8.9125099, 5.4432116, -13.7713280, 14.0206385
9: -5.8088717, 5.9636827, -6.2058897, 6.3725805, -12.1814499, 12.1695728

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418394, upper bound: 10.8418102
time: 2.79 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418103
time: 3.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1149702, 4.1899500, -7.8389311, 4.0316324, -12.1466026, 12.0288811
1: -5.3421764, 5.1266422, -5.1678519, 4.9681892, -10.3103638, 10.2944927
2: -6.9299612, 5.2153306, -6.6845655, 5.0431991, -11.9731588, 11.8998966
3: -7.3272820, 4.5856667, -7.0717106, 4.4397602, -11.7670422, 11.6573772
4: -7.9836326, 6.3921490, -7.7241654, 6.1835432, -14.1671753, 14.1163139
5: -6.6160822, 4.7216988, -6.3878245, 4.5578728, -11.1739540, 11.1095219
6: -6.1771340, 6.2783313, -5.9504004, 6.0593309, -12.2364635, 12.2287312
7: -6.8047247, 6.2653217, -6.5693750, 6.0555334, -12.8602581, 12.8346968
8: -8.6719351, 5.3173971, -8.3788776, 5.1373258, -13.8092613, 13.6962748
9: -6.0389681, 6.2054257, -5.8431759, 5.9990039, -12.0379715, 12.0486012

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 70

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418356, upper bound: 10.8418095
time: 3.23 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418101, upper bound: 10.8418097
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.1543789, 4.1737638, -9.6746130, 4.8597627, -13.0141411, 13.8483772
1: -5.3858962, 5.1773233, -6.4227462, 6.1590152, -11.5449114, 11.6000690
2: -6.9827189, 5.2423444, -8.3960371, 6.1954660, -13.1781845, 13.6383820
3: -7.3986549, 4.6178269, -8.9504013, 5.4612012, -12.8598557, 13.5682278
4: -8.0726643, 6.4456978, -9.7094669, 7.6931381, -15.7658024, 16.1551647
5: -6.6597080, 4.7296515, -7.9560471, 5.5619626, -12.2216702, 12.6856985
6: -6.2019081, 6.3079596, -7.4066987, 7.4998121, -13.7017202, 13.7146587
7: -6.8531551, 6.3005705, -8.1930113, 7.4631186, -14.3162737, 14.4935818
8: -8.7444029, 5.3420844, -10.4771242, 6.3269091, -15.0713120, 15.8192081
9: -6.0936799, 6.2546110, -7.2737603, 7.4644775, -13.5581570, 13.5283718

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418399, upper bound: 10.8418131
time: 3.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418399, upper bound: 10.8418473
time: 3.28 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.4582253, 4.7511587, -8.0288315, 4.1115885, -13.5698137, 12.7799902
1: -6.2819653, 6.0279970, -5.3037448, 5.1002073, -11.3821726, 11.3317413
2: -8.2018509, 6.0607939, -6.8692131, 5.1641765, -13.3660278, 12.9300070
3: -8.7417545, 5.3448906, -7.2769246, 4.5498085, -13.2915630, 12.6218147
4: -9.4940214, 7.5237131, -7.9460616, 6.3469553, -15.8409767, 15.4697742
5: -7.7768173, 5.4379411, -6.5558634, 4.6576014, -12.4344187, 11.9938049
6: -7.2324715, 7.3299489, -6.0997963, 6.2089782, -13.4414501, 13.4297447
7: -8.0067978, 7.3013678, -6.7443414, 6.2058315, -14.2126293, 14.0457096
8: -10.2411385, 6.1869874, -8.6065540, 5.2607832, -15.5019217, 14.7935410
9: -7.1145487, 7.2988954, -6.0006413, 6.1578093, -13.2723579, 13.2995367

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417850, upper bound: 10.8418321
time: 3.03 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417855, upper bound: 10.8418168
time: 2.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.67 seconds
IS_A1_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8418394, upper bound: 10.8418102
IS_A1_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418103
IS_A1_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8418356, upper bound: 10.8418095
IS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8418101, upper bound: 10.8418097
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8418399, upper bound: 10.8418131
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8418399, upper bound: 10.8418473
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8417850, upper bound: 10.8418321
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.67
Output dim: 0, lower bound: -10.8417855, upper bound: 10.8418168

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1543789, 4.1737638, -9.4582253, 4.7511587, -12.9055376, 13.6319885
1: -5.3858962, 5.1773233, -6.2819653, 6.0279970, -11.4138927, 11.4592886
2: -6.9827189, 5.2423444, -8.2018509, 6.0607939, -13.0435123, 13.4441948
3: -7.3986549, 4.6178269, -8.7417545, 5.3448906, -12.7435455, 13.3595810
4: -8.0726643, 6.4456978, -9.4940214, 7.5237131, -15.5963774, 15.9397192
5: -6.6597080, 4.7296515, -7.7768173, 5.4379411, -12.0976486, 12.5064688
6: -6.2019081, 6.3079596, -7.2324715, 7.3299489, -13.5318565, 13.5404310
7: -6.8531551, 6.3005705, -8.0067978, 7.3013678, -14.1545229, 14.3073683
8: -8.7444029, 5.3420844, -10.2411385, 6.1869874, -14.9313908, 15.5832233
9: -6.0936799, 6.2546110, -7.1145487, 7.2988954, -13.3925753, 13.3691597

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 70

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417293, upper bound: 10.8417785
time: 4.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418178, upper bound: 10.8418159
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418154, upper bound: 10.8418184
time: 4.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.73 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 0, lower bound: -10.8418178, upper bound: 10.8418159
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 0, lower bound: -10.8418154, upper bound: 10.8418184
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=13.22586727142334
rel_dist={0: [-10.84189901486047, 10.841897299947618]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418508
time: 3.85 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418504
time: 2.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.33
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418508
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.33
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418504

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -8.7459450, 4.4799228, -12.9330826, 13.0691080
1: -5.5820465, 5.3603401, -5.7760954, 5.5418692, -11.1239147, 11.1364355
2: -7.2538810, 5.4287362, -7.5248165, 5.6160679, -12.8699474, 12.9535513
3: -7.6901236, 4.7797179, -7.9764252, 4.9407926, -12.6309166, 12.7561417
4: -8.3745241, 6.6817288, -8.6739941, 6.9140596, -15.2885809, 15.3557224
5: -6.9095788, 4.9000840, -7.1571879, 5.0683022, -11.9778805, 12.0572701
6: -6.4428177, 6.5433078, -6.6823978, 6.7787600, -13.2215748, 13.2257061
7: -7.1115313, 6.5268288, -7.3757725, 6.7552648, -13.8667936, 13.9026012
8: -9.0736485, 5.5359268, -9.3987417, 5.7281442, -14.8017912, 14.9346676
9: -6.3155327, 6.4854898, -6.5382051, 6.7157540, -13.0312862, 13.0236950

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417686, upper bound: 10.8417802
time: 2.75 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417923, upper bound: 10.8417964
time: 2.99 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418504
time: 2.24 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418504
time: 2.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7146721, 4.8798447, -8.4372864, 4.3178682, -14.0325403, 13.3171310
1: -6.4488497, 6.1833115, -5.5718665, 5.3509436, -11.7997932, 11.7551765
2: -8.4320307, 6.2204180, -7.2404470, 5.4199591, -13.8519897, 13.4608650
3: -8.9891014, 5.4827590, -7.6749759, 4.7716808, -13.7607822, 13.1577330
4: -9.7494392, 7.7245536, -8.3587685, 6.6695857, -16.4190254, 16.0833206
5: -7.9892755, 5.5848994, -6.8976059, 4.8911552, -12.8804302, 12.4825048
6: -7.4389572, 7.5312815, -6.4301300, 6.5316191, -13.9705763, 13.9614105
7: -8.2275219, 7.4930902, -7.1002474, 6.5159197, -14.7434387, 14.5933380
8: -10.5208874, 6.3528190, -9.0570726, 5.5256128, -16.0465012, 15.4098911
9: -7.3032880, 7.4951725, -6.3050575, 6.4743299, -13.7776184, 13.8002300

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418507, upper bound: 10.8418487
time: 3.25 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418504
time: 3.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.91 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.91
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418504
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.91
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418504
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.91
Output dim: 0, lower bound: -10.8418507, upper bound: 10.8418487
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.91
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418504

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -8.4531593, 4.3231654, -12.7763233, 12.7763243
1: -5.5820465, 5.3603401, -5.5820465, 5.3603401, -10.9423857, 10.9423857
2: -7.2538810, 5.4287362, -7.2538810, 5.4287362, -12.6826153, 12.6826153
3: -7.6901236, 4.7797179, -7.6901236, 4.7797179, -12.4698410, 12.4698410
4: -8.3745241, 6.6817288, -8.3745241, 6.6817288, -15.0562515, 15.0562506
5: -6.9095788, 4.9000840, -6.9095788, 4.9000840, -11.8096619, 11.8096619
6: -6.4428177, 6.5433078, -6.4428177, 6.5433078, -12.9861259, 12.9861259
7: -7.1115313, 6.5268288, -7.1115313, 6.5268288, -13.6383591, 13.6383591
8: -9.0736485, 5.5359268, -9.0736485, 5.5359268, -14.6095715, 14.6095734
9: -6.3155327, 6.4854898, -6.3155327, 6.4854898, -12.8010225, 12.8010225

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417683, upper bound: 10.8417804
time: 3.28 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417907, upper bound: 10.8417973
time: 3.07 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417533, upper bound: 10.8417646
time: 3.08 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418429, upper bound: 10.8418393
time: 2.92 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418736, upper bound: 10.8418486
time: 3.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.4531593, 4.3231654, -9.7146721, 4.8798447, -13.3330040, 14.0378361
1: -5.5820465, 5.3603401, -6.4488497, 6.1833115, -11.7653580, 11.8091898
2: -7.2538810, 5.4287362, -8.4320307, 6.2204180, -13.4742966, 13.8607674
3: -7.6901236, 4.7797179, -8.9891014, 5.4827590, -13.1728821, 13.7688198
4: -8.3745241, 6.6817288, -9.7494392, 7.7245536, -16.0990753, 16.4311676
5: -6.9095788, 4.9000840, -7.9892755, 5.5848994, -12.4944782, 12.8893595
6: -6.4428177, 6.5433078, -7.4389572, 7.5312815, -13.9740982, 13.9822655
7: -7.1115313, 6.5268288, -8.2275219, 7.4930902, -14.6046219, 14.7543497
8: -9.0736485, 5.5359268, -10.5208874, 6.3528190, -15.4264679, 16.0568142
9: -6.3155327, 6.4854898, -7.3032880, 7.4951725, -13.8107052, 13.7887783

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417683, upper bound: 10.8417802
time: 3.54 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418753, upper bound: 10.8418507
time: 2.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418753, upper bound: 10.8418508
time: 3.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.0589199, 4.5569534, -6.7683640, 3.4979184, -12.5568380, 11.3253174
1: -6.0206680, 5.7805281, -4.4901118, 4.3273430, -10.3480110, 10.2706394
2: -7.8431592, 5.8143358, -5.7453694, 4.3933153, -12.2364750, 11.5597057
3: -8.3558092, 5.1278992, -6.0804515, 3.8703341, -12.2261429, 11.2083511
4: -9.0860157, 7.2117500, -6.6805182, 5.3685303, -14.4545460, 13.8922682
5: -7.4502993, 5.2105031, -5.5276823, 3.9364762, -11.3867760, 10.7381859
6: -6.9066586, 7.0193729, -5.0722246, 5.2325134, -12.1391716, 12.0915976
7: -7.6615243, 7.0014200, -5.6714387, 5.2665205, -12.9280453, 12.6728592
8: -9.8061886, 5.9277945, -7.2421284, 4.4474945, -14.2536831, 13.1699228
9: -6.8183026, 6.9929461, -5.0776315, 5.2038164, -12.0221195, 12.0705776

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418480
time: 2.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418473
time: 2.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.6062813, 4.8261495, -7.8422079, 4.0236340, -13.6299152, 12.6683578
1: -6.3779860, 6.1166954, -5.1816640, 4.9830322, -11.3610182, 11.2983589
2: -8.3345385, 6.1531363, -6.7020245, 5.0506220, -13.3851604, 12.8551607
3: -8.8842602, 5.4240389, -7.0963960, 4.4479938, -13.3322544, 12.5204353
4: -9.6396961, 7.6396809, -7.7531271, 6.2014952, -15.8411913, 15.3928080
5: -7.8999367, 5.5230165, -6.4041390, 4.5517745, -12.4517117, 11.9271555
6: -7.3509569, 7.4465227, -5.9466391, 6.0652723, -13.4162292, 13.3931618
7: -8.1337309, 7.4116693, -6.5850921, 6.0658083, -14.1995392, 13.9967613
8: -10.4025497, 6.2825413, -8.4033041, 5.1398892, -15.5424385, 14.6858454
9: -7.2229438, 7.4120140, -5.8624129, 6.0153284, -13.2382717, 13.2744274

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418473, upper bound: 10.8418498
time: 3.18 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418473, upper bound: 10.8418510
time: 2.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 11.25 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418429, upper bound: 10.8418393
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418736, upper bound: 10.8418486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418753, upper bound: 10.8418507
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418753, upper bound: 10.8418508
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418480
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418473
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418473, upper bound: 10.8418498
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.25
Output dim: 0, lower bound: -10.8418473, upper bound: 10.8418510

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7710667, 4.4511189, -8.1057243, 4.1502533, -12.9213200, 12.5568428
1: -5.8131151, 5.5860500, -5.3522854, 5.1460314, -10.9591465, 10.9383354
2: -7.5656533, 5.6316767, -6.9363565, 5.2108183, -12.7764721, 12.5680332
3: -8.0398540, 4.9661331, -7.3480644, 4.5907068, -12.6305609, 12.3141975
4: -8.7556324, 6.9592280, -8.0207186, 6.4048576, -15.1604900, 14.9799461
5: -7.1914692, 5.0650182, -6.6169243, 4.7026100, -11.8940792, 11.6819420
6: -6.6938596, 6.7939858, -6.1632748, 6.2684059, -12.9622650, 12.9572601
7: -7.4068289, 6.7790213, -6.8090453, 6.2626557, -13.6694851, 13.5880661
8: -9.4592066, 5.7437124, -8.6874914, 5.3104811, -14.7696877, 14.4312038
9: -6.5829935, 6.7539654, -6.0552711, 6.2149410, -12.7979345, 12.8092365

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418114, upper bound: 10.8418481
time: 3.19 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418301
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.1543789, 4.1737638, -8.4531593, 4.3231654, -12.4775429, 12.6269226
1: -5.3858962, 5.1773233, -5.5820465, 5.3603401, -10.7462368, 10.7593679
2: -6.9827189, 5.2423444, -7.2538810, 5.4287362, -12.4114552, 12.4962254
3: -7.3986549, 4.6178269, -7.6901236, 4.7797179, -12.1783733, 12.3079510
4: -8.0726643, 6.4456978, -8.3745241, 6.6817288, -14.7543926, 14.8202219
5: -6.6597080, 4.7296515, -6.9095788, 4.9000840, -11.5597916, 11.6392307
6: -6.2019081, 6.3079596, -6.4428177, 6.5433078, -12.7452164, 12.7507744
7: -6.8531551, 6.3005705, -7.1115313, 6.5268288, -13.3799839, 13.4121017
8: -8.7444029, 5.3420844, -9.0736485, 5.5359268, -14.2803288, 14.4157333
9: -6.0936799, 6.2546110, -6.3155327, 6.4854898, -12.5791693, 12.5701437

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418896, upper bound: 10.8418688
time: 4.34 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418896, upper bound: 10.8418963
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.7842145, 3.5019553, -9.0589199, 4.5569534, -11.3411674, 12.5608749
1: -4.4997196, 4.3365183, -6.0206680, 5.7805281, -10.2802477, 10.3571863
2: -5.7580748, 4.4015784, -7.8431592, 5.8143358, -11.5724106, 12.2447376
3: -6.0936804, 3.8780322, -8.3558092, 5.1278992, -11.2215796, 12.2338409
4: -6.6953669, 5.3803387, -9.0860157, 7.2117500, -13.9071169, 14.4663544
5: -5.5390244, 3.9451072, -7.4502993, 5.2105031, -10.7495270, 11.3954067
6: -5.0846615, 5.2438340, -6.9066586, 7.0193729, -12.1040344, 12.1504927
7: -5.6818085, 5.2764645, -7.6615243, 7.0014200, -12.6832285, 12.9379883
8: -7.2579589, 4.4572582, -9.8061886, 5.9277945, -13.1857529, 14.2634468
9: -5.0875573, 5.2140598, -6.8183026, 6.9929461, -12.0805035, 12.0323620

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417913, upper bound: 10.8417944
time: 2.84 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418735, upper bound: 10.8418469
time: 4.03 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418735, upper bound: 10.8418507
time: 4.18 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.8623919, 4.0305882, -9.6062813, 4.8261495, -12.6885414, 13.6368694
1: -5.1942968, 4.9949150, -6.3779860, 6.1166954, -11.3109922, 11.3729010
2: -6.7187901, 5.0618858, -8.3345385, 6.1531363, -12.8719263, 13.3964243
3: -7.1151147, 4.4582224, -8.8842602, 5.4240389, -12.5391541, 13.3424826
4: -7.7725391, 6.2167931, -9.6396961, 7.6396809, -15.4122200, 15.8564892
5: -6.4192629, 4.5632172, -7.8999367, 5.5230165, -11.9422798, 12.4631538
6: -5.9627509, 6.0802150, -7.3509569, 7.4465227, -13.4092731, 13.4311714
7: -6.5997624, 6.0794239, -8.1337309, 7.4116693, -14.0114317, 14.2131548
8: -8.4242134, 5.1527052, -10.4025497, 6.2825413, -14.7067547, 15.5552549
9: -5.8758101, 6.0295091, -7.2229438, 7.4120140, -13.2878246, 13.2524529

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418476
time: 4.49 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418497
time: 3.96 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.9389896, 4.0078292, -6.7683640, 3.4979184, -11.4369078, 10.7761936
1: -5.2842102, 5.0867510, -4.4901118, 4.3273430, -9.6115532, 9.5768623
2: -6.8300972, 5.1184130, -5.7453694, 4.3933153, -11.2234125, 10.8637829
3: -7.2633328, 4.5182614, -6.0804515, 3.8703341, -11.1336670, 10.5987129
4: -7.9412947, 6.3284316, -6.6805182, 5.3685303, -13.3098249, 13.0089493
5: -6.5214534, 4.5743227, -5.5276823, 3.9364762, -10.4579296, 10.1020050
6: -6.0010281, 6.1445255, -5.0722246, 5.2325134, -11.2335415, 11.2167501
7: -6.6886630, 6.1578865, -5.6714387, 5.2665205, -11.9551830, 11.8293247
8: -8.5742321, 5.2025962, -7.2421284, 4.4474945, -13.0217266, 12.4447250
9: -5.9821663, 6.1277628, -5.0776315, 5.2038164, -11.1859827, 11.2053947

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418479
time: 2.18 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418469, upper bound: 10.8418477
time: 2.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1093225, 4.5797076, -6.7683640, 3.4979184, -12.6072407, 11.3480721
1: -6.0533729, 5.8115520, -4.4901118, 4.3273430, -10.3807163, 10.3016644
2: -7.8877387, 5.8447852, -5.7453694, 4.3933153, -12.2810535, 11.5901546
3: -8.4040852, 5.1549087, -6.0804515, 3.8703341, -12.2744198, 11.2353601
4: -9.1370716, 7.2508297, -6.6805182, 5.3685303, -14.5056019, 13.9313478
5: -7.4905796, 5.2391715, -5.5276823, 3.9364762, -11.4270554, 10.7668533
6: -6.9476757, 7.0581341, -5.0722246, 5.2325134, -12.1801891, 12.1303587
7: -7.7037601, 7.0385532, -5.6714387, 5.2665205, -12.9702806, 12.7099915
8: -9.8604088, 5.9603915, -7.2421284, 4.4474945, -14.3079033, 13.2025204
9: -6.8548994, 7.0309110, -5.0776315, 5.2038164, -12.0587158, 12.1085424

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 70

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418307, upper bound: 10.8418059
time: 2.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418456
time: 3.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.9389896, 4.0078292, -7.8422079, 4.0236340, -11.9626236, 11.8500366
1: -5.2842102, 5.0867510, -5.1816640, 4.9830322, -10.2672424, 10.2684155
2: -6.8300972, 5.1184130, -6.7020245, 5.0506220, -11.8807192, 11.8204374
3: -7.2633328, 4.5182614, -7.0963960, 4.4479938, -11.7113266, 11.6146574
4: -7.9412947, 6.3284316, -7.7531271, 6.2014952, -14.1427898, 14.0815582
5: -6.5214534, 4.5743227, -6.4041390, 4.5517745, -11.0732279, 10.9784622
6: -6.0010281, 6.1445255, -5.9466391, 6.0652723, -12.0663004, 12.0911646
7: -6.6886630, 6.1578865, -6.5850921, 6.0658083, -12.7544708, 12.7429790
8: -8.5742321, 5.2025962, -8.4033041, 5.1398892, -13.7141209, 13.6058998
9: -5.9821663, 6.1277628, -5.8624129, 6.0153284, -11.9974947, 11.9901752

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418468, upper bound: 10.8418510
time: 3.18 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418482, upper bound: 10.8418500
time: 3.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1093225, 4.5797076, -7.8422079, 4.0236340, -13.1329565, 12.4219151
1: -6.0533729, 5.8115520, -5.1816640, 4.9830322, -11.0364056, 10.9932156
2: -7.8877387, 5.8447852, -6.7020245, 5.0506220, -12.9383602, 12.5468102
3: -8.4040852, 5.1549087, -7.0963960, 4.4479938, -12.8520794, 12.2513046
4: -9.1370716, 7.2508297, -7.7531271, 6.2014952, -15.3385668, 15.0039568
5: -7.4905796, 5.2391715, -6.4041390, 4.5517745, -12.0423546, 11.6433105
6: -6.9476757, 7.0581341, -5.9466391, 6.0652723, -13.0129480, 13.0047731
7: -7.7037601, 7.0385532, -6.5850921, 6.0658083, -13.7695684, 13.6236458
8: -9.8604088, 5.9603915, -8.4033041, 5.1398892, -15.0002975, 14.3636951
9: -6.8548994, 7.0309110, -5.8624129, 6.0153284, -12.8702278, 12.8933239

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418309, upper bound: 10.8418129
time: 2.72 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418486
time: 2.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.65 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418114, upper bound: 10.8418481
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418301
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418896, upper bound: 10.8418688
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418896, upper bound: 10.8418963
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418735, upper bound: 10.8418469
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418735, upper bound: 10.8418507
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418476
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418497
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418479
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418469, upper bound: 10.8418477
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418307, upper bound: 10.8418059
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418456
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418468, upper bound: 10.8418510
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418482, upper bound: 10.8418500
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418309, upper bound: 10.8418129
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.65
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418486

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7710667, 4.4511189, -7.6591330, 3.9260876, -12.6971540, 12.1102524
1: -5.8131151, 5.5860500, -5.0632014, 4.8733950, -10.6865101, 10.6492519
2: -7.5656533, 5.6316767, -6.5360188, 4.9349399, -12.5005932, 12.1676960
3: -8.0398540, 4.9661331, -6.9208140, 4.3495193, -12.3893738, 11.8869476
4: -8.7556324, 6.9592280, -7.5745711, 6.0572433, -14.8128757, 14.5337992
5: -7.1914692, 5.0650182, -6.2497611, 4.4455156, -11.6369848, 11.3147793
6: -6.6938596, 6.7939858, -5.7993450, 5.9189906, -12.6128502, 12.5933304
7: -7.4068289, 6.7790213, -6.4251294, 5.9270549, -13.3338833, 13.2041512
8: -9.4592066, 5.7437124, -8.2019653, 5.0215921, -14.4807987, 13.9456778
9: -6.5829935, 6.7539654, -5.7269654, 5.8741207, -12.4571142, 12.4809303

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417071, upper bound: 10.8417081
time: 2.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418122, upper bound: 10.8418400
time: 2.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418122, upper bound: 10.8418501
time: 2.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.1543789, 4.1737638, -8.7710667, 4.4511189, -12.6054974, 12.9448299
1: -5.3858962, 5.1773233, -5.8131151, 5.5860500, -10.9719467, 10.9904385
2: -6.9827189, 5.2423444, -7.5656533, 5.6316767, -12.6143951, 12.8079977
3: -7.3986549, 4.6178269, -8.0398540, 4.9661331, -12.3647881, 12.6576805
4: -8.0726643, 6.4456978, -8.7556324, 6.9592280, -15.0318928, 15.2013302
5: -6.6597080, 4.7296515, -7.1914692, 5.0650182, -11.7247257, 11.9211206
6: -6.2019081, 6.3079596, -6.6938596, 6.7939858, -12.9958935, 13.0018196
7: -6.8531551, 6.3005705, -7.4068289, 6.7790213, -13.6321764, 13.7073994
8: -8.7444029, 5.3420844, -9.4592066, 5.7437124, -14.4881153, 14.8012905
9: -6.0936799, 6.2546110, -6.5829935, 6.7539654, -12.8476448, 12.8376045

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418117
time: 3.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418322, upper bound: 10.8418124
time: 5.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.1543789, 4.1737638, -8.1543789, 4.1737638, -12.3281422, 12.3281422
1: -5.3858962, 5.1773233, -5.3858962, 5.1773233, -10.5632191, 10.5632191
2: -6.9827189, 5.2423444, -6.9827189, 5.2423444, -12.2250633, 12.2250633
3: -7.3986549, 4.6178269, -7.3986549, 4.6178269, -12.0164814, 12.0164814
4: -8.0726643, 6.4456978, -8.0726643, 6.4456978, -14.5183620, 14.5183620
5: -6.6597080, 4.7296515, -6.6597080, 4.7296515, -11.3893595, 11.3893595
6: -6.2019081, 6.3079596, -6.2019081, 6.3079596, -12.5098677, 12.5098677
7: -6.8531551, 6.3005705, -6.8531551, 6.3005705, -13.1537256, 13.1537256
8: -8.7444029, 5.3420844, -8.7444029, 5.3420844, -14.0864868, 14.0864868
9: -6.0936799, 6.2546110, -6.0936799, 6.2546110, -12.3482914, 12.3482914

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418111, upper bound: 10.8418035
time: 2.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417812, upper bound: 10.8418009
time: 3.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.7842145, 3.5019553, -7.9389896, 4.0078292, -10.7920437, 11.4409447
1: -4.4997196, 4.3365183, -5.2842102, 5.0867510, -9.5864706, 9.6207285
2: -5.7580748, 4.4015784, -6.8300972, 5.1184130, -10.8764877, 11.2316761
3: -6.0936804, 3.8780322, -7.2633328, 4.5182614, -10.6119423, 11.1413651
4: -6.6953669, 5.3803387, -7.9412947, 6.3284316, -13.0237980, 13.3216333
5: -5.5390244, 3.9451072, -6.5214534, 4.5743227, -10.1133471, 10.4665604
6: -5.0846615, 5.2438340, -6.0010281, 6.1445255, -11.2291870, 11.2448616
7: -5.6818085, 5.2764645, -6.6886630, 6.1578865, -11.8396950, 11.9651279
8: -7.2579589, 4.4572582, -8.5742321, 5.2025962, -12.4605551, 13.0314903
9: -5.0875573, 5.2140598, -5.9821663, 6.1277628, -11.2153206, 11.1962261

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418481
time: 4.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418729, upper bound: 10.8418472
time: 2.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.7842145, 3.5019553, -9.1093225, 4.5797076, -11.3639221, 12.6112776
1: -4.4997196, 4.3365183, -6.0533729, 5.8115520, -10.3112717, 10.3898907
2: -5.7580748, 4.4015784, -7.8877387, 5.8447852, -11.6028595, 12.2893171
3: -6.0936804, 3.8780322, -8.4040852, 5.1549087, -11.2485886, 12.2821178
4: -6.6953669, 5.3803387, -9.1370716, 7.2508297, -13.9461966, 14.5174103
5: -5.5390244, 3.9451072, -7.4905796, 5.2391715, -10.7781963, 11.4356871
6: -5.0846615, 5.2438340, -6.9476757, 7.0581341, -12.1427956, 12.1915092
7: -5.6818085, 5.2764645, -7.7037601, 7.0385532, -12.7203617, 12.9802246
8: -7.2579589, 4.4572582, -9.8604088, 5.9603915, -13.2183504, 14.3176670
9: -5.0875573, 5.2140598, -6.8548994, 7.0309110, -12.1184683, 12.0689592

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417902, upper bound: 10.8417951
time: 3.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418374
time: 3.31 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418734, upper bound: 10.8418495
time: 5.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.8623919, 4.0305882, -7.9389896, 4.0078292, -11.8702211, 11.9695778
1: -5.1942968, 4.9949150, -5.2842102, 5.0867510, -10.2810478, 10.2791252
2: -6.7187901, 5.0618858, -6.8300972, 5.1184130, -11.8372030, 11.8919830
3: -7.1151147, 4.4582224, -7.2633328, 4.5182614, -11.6333761, 11.7215557
4: -7.7725391, 6.2167931, -7.9412947, 6.3284316, -14.1009712, 14.1580877
5: -6.4192629, 4.5632172, -6.5214534, 4.5743227, -10.9935856, 11.0846710
6: -5.9627509, 6.0802150, -6.0010281, 6.1445255, -12.1072769, 12.0812435
7: -6.5997624, 6.0794239, -6.6886630, 6.1578865, -12.7576485, 12.7680874
8: -8.4242134, 5.1527052, -8.5742321, 5.2025962, -13.6268101, 13.7269373
9: -5.8758101, 6.0295091, -5.9821663, 6.1277628, -12.0035725, 12.0116749

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418734, upper bound: 10.8418483
time: 3.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418726, upper bound: 10.8418466
time: 3.25 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.8623919, 4.0305882, -9.1093225, 4.5797076, -12.4420996, 13.1399107
1: -5.1942968, 4.9949150, -6.0533729, 5.8115520, -11.0058489, 11.0482883
2: -6.7187901, 5.0618858, -7.8877387, 5.8447852, -12.5635757, 12.9496250
3: -7.1151147, 4.4582224, -8.4040852, 5.1549087, -12.2700233, 12.8623075
4: -7.7725391, 6.2167931, -9.1370716, 7.2508297, -15.0233688, 15.3538647
5: -6.4192629, 4.5632172, -7.4905796, 5.2391715, -11.6584339, 12.0537968
6: -5.9627509, 6.0802150, -6.9476757, 7.0581341, -13.0208855, 13.0278912
7: -6.5997624, 6.0794239, -7.7037601, 7.0385532, -13.6383152, 13.7831841
8: -8.4242134, 5.1527052, -9.8604088, 5.9603915, -14.3846054, 15.0131140
9: -5.8758101, 6.0295091, -6.8548994, 7.0309110, -12.9067211, 12.8844090

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418400
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418726, upper bound: 10.8418477
time: 4.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.7609200, 3.9201691, -5.8475008, 3.0516741, -10.8125944, 9.7676697
1: -5.1652188, 4.9742346, -3.8799965, 3.7482939, -8.9135132, 8.8542309
2: -6.6658468, 5.0065084, -4.8986607, 3.8176644, -10.4835110, 9.9051685
3: -7.0862846, 4.4198561, -5.2159638, 3.3649223, -10.4512072, 9.6358204
4: -7.7552509, 6.1850510, -5.7268906, 4.6285658, -12.3838167, 11.9119415
5: -6.3697948, 4.4747143, -4.7478170, 3.4258020, -9.7955971, 9.2225313
6: -5.8586807, 6.0046082, -4.3340011, 4.5143423, -10.3730230, 10.3386097
7: -6.5311246, 6.0210676, -4.8603268, 4.5704370, -11.1015615, 10.8813944
8: -8.3739367, 5.0874991, -6.2132330, 3.8586020, -12.2325382, 11.3007317
9: -5.8456268, 5.9873142, -4.3758717, 4.4964042, -10.3420315, 10.3631859

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418191, upper bound: 10.8418271
time: 3.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418193
time: 2.22 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.8720102, 3.9749968, -6.4199109, 3.3288748, -11.2008848, 10.3949080
1: -5.2393742, 5.0443249, -4.2591209, 4.1077271, -9.3471012, 9.3034458
2: -6.7682114, 5.0762949, -5.4249730, 4.1751776, -10.9433889, 10.5012684
3: -7.1966228, 4.4812117, -5.7528505, 3.6785765, -10.8751993, 10.2340622
4: -7.8711338, 6.2744088, -6.3182516, 5.0886574, -12.9597912, 12.5926609
5: -6.4644246, 4.5368576, -5.2322655, 3.7437067, -10.2081318, 9.7691231
6: -5.9474192, 6.0918155, -4.7930822, 4.9613447, -10.9087639, 10.8848972
7: -6.6293244, 6.1063948, -5.3641958, 5.0028486, -11.6321735, 11.4705906
8: -8.4987736, 5.1592588, -6.8525772, 4.2247682, -12.7235413, 12.0118361
9: -5.9307365, 6.0748653, -4.8115325, 4.9355879, -10.8663244, 10.8863983

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418183, upper bound: 10.8418260
time: 2.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418187
time: 2.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1093225, 4.5797076, -6.4561996, 3.3463447, -12.4556675, 11.0359077
1: -6.0533729, 5.8115520, -4.2862635, 4.1352739, -10.1886463, 10.0978155
2: -7.8877387, 5.8447852, -5.4623518, 4.1997089, -12.0874481, 11.3071365
3: -8.4040852, 5.1549087, -5.7924414, 3.7012844, -12.1053696, 10.9473495
4: -9.1370716, 7.2508297, -6.3652387, 5.1205397, -14.2576113, 13.6160679
5: -7.4905796, 5.2391715, -5.2659602, 3.7619736, -11.2525530, 10.5051317
6: -6.9476757, 7.0581341, -4.8225694, 4.9895964, -11.9372721, 11.8807030
7: -7.7037601, 7.0385532, -5.4021320, 5.0342255, -12.7379856, 12.4406853
8: -9.8604088, 5.9603915, -6.8978853, 4.2486434, -14.1090527, 12.8582764
9: -6.8548994, 7.0309110, -4.8445997, 4.9680042, -11.8229036, 11.8755112

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418478, upper bound: 10.8418455
time: 3.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418464
time: 2.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.7609200, 3.9201691, -6.9403009, 3.5794787, -11.3403988, 10.8604698
1: -5.1652188, 4.9742346, -4.5879822, 4.4214373, -9.5866566, 9.5622168
2: -6.6658468, 5.0065084, -5.8799038, 4.4878955, -11.1537418, 10.8864117
3: -7.0862846, 4.4198561, -6.2141051, 3.9554801, -11.0417652, 10.6339607
4: -7.7552509, 6.1850510, -6.8298469, 5.4842987, -13.2395496, 13.0148983
5: -6.3697948, 4.4747143, -5.6473989, 4.0413370, -10.4111319, 10.1221132
6: -5.8586807, 6.0046082, -5.2206135, 5.3583403, -11.2170210, 11.2252216
7: -6.5311246, 6.0210676, -5.7967434, 5.3802752, -11.9113998, 11.8178110
8: -8.3739367, 5.0874991, -7.4041095, 4.5562878, -12.9302244, 12.4916086
9: -5.8456268, 5.9873142, -5.1832180, 5.3143883, -11.1600151, 11.1705322

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418280
time: 2.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418220
time: 2.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.8720102, 3.9749968, -7.4764814, 3.8447301, -11.7167406, 11.4514780
1: -5.2393742, 5.0443249, -4.9385862, 4.7524753, -9.9918499, 9.9829111
2: -6.7682114, 5.0762949, -6.3659792, 4.8215580, -11.5897694, 11.4422741
3: -7.1966228, 4.4812117, -6.7340775, 4.2468243, -11.4434471, 11.2152891
4: -7.8711338, 6.2744088, -7.3729010, 5.9079800, -13.7791138, 13.6473103
5: -6.4644246, 4.5368576, -6.0942822, 4.3466616, -10.8110867, 10.6311398
6: -5.9474192, 6.0918155, -5.6536388, 5.7784953, -11.7259140, 11.7454548
7: -6.6293244, 6.1063948, -6.2630014, 5.7859631, -12.4152870, 12.3693962
8: -8.4987736, 5.1592588, -7.9939709, 4.9033256, -13.4020996, 13.1532297
9: -5.9307365, 6.0748653, -5.5834131, 5.7283025, -11.6590385, 11.6582785

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418176, upper bound: 10.8418267
time: 3.13 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418191, upper bound: 10.8418209
time: 3.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1093225, 4.5797076, -7.5353460, 3.8715367, -12.9808598, 12.1150532
1: -6.0533729, 5.8115520, -4.9816504, 4.7951336, -10.8485069, 10.7932024
2: -7.8877387, 5.8447852, -6.4250069, 4.8600698, -12.7478085, 12.2697926
3: -8.4040852, 5.1549087, -6.7991381, 4.2821770, -12.6862621, 11.9540462
4: -9.1370716, 7.2508297, -7.4452305, 5.9592419, -15.0963135, 14.6960602
5: -7.4905796, 5.2391715, -6.1479549, 4.3769145, -11.8674946, 11.3871269
6: -6.9476757, 7.0581341, -5.7000742, 5.8246045, -12.7722797, 12.7582083
7: -7.7037601, 7.0385532, -6.3207989, 5.8349400, -13.5387001, 13.3593521
8: -9.8604088, 5.9603915, -8.0657997, 4.9419928, -14.8024015, 14.0261917
9: -6.8548994, 7.0309110, -5.6347437, 5.7792683, -12.6341677, 12.6656551

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 70

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418113, upper bound: 10.8418346
time: 2.29 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418119, upper bound: 10.8418480
time: 2.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 10.15 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418122, upper bound: 10.8418400
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418122, upper bound: 10.8418501
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418117
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418322, upper bound: 10.8418124
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418111, upper bound: 10.8418035
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8417812, upper bound: 10.8418009
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418481
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418729, upper bound: 10.8418472
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418374
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418734, upper bound: 10.8418495
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418734, upper bound: 10.8418483
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418726, upper bound: 10.8418466
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418400
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418726, upper bound: 10.8418477
IS_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418191, upper bound: 10.8418271
IS_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418193
IS_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418183, upper bound: 10.8418260
IS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418187
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418478, upper bound: 10.8418455
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418483, upper bound: 10.8418464
IS_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418280
IS_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418220
IS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418176, upper bound: 10.8418267
IS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418191, upper bound: 10.8418209
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418113, upper bound: 10.8418346
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.15
Output dim: 0, lower bound: -10.8418119, upper bound: 10.8418480

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.7710667, 4.4511189, -8.3535156, 4.2414622, -13.0125294, 12.8046341
1: -5.8131151, 5.5860500, -5.5427141, 5.3322773, -11.1453924, 11.1287642
2: -7.5656533, 5.6316767, -7.1935520, 5.3739529, -12.9396057, 12.8252287
3: -8.0398540, 4.9661331, -7.6416631, 4.7404404, -12.7802944, 12.6077957
4: -8.7556324, 6.9592280, -8.3381834, 6.6367073, -15.3923397, 15.2974110
5: -7.1914692, 5.0650182, -6.8516784, 4.8238721, -12.0153408, 11.9166965
6: -6.6938596, 6.7939858, -6.3519821, 6.4674263, -13.1612854, 13.1459675
7: -7.4068289, 6.7790213, -7.0485158, 6.4663334, -13.8731623, 13.8275375
8: -9.4592066, 5.7437124, -9.0084543, 5.4729772, -14.9321842, 14.7521667
9: -6.5829935, 6.7539654, -6.2780080, 6.4362836, -13.0192776, 13.0319729

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8416564, upper bound: 10.8416535
time: 2.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418395
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418406
time: 2.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.7710667, 4.4511189, -7.6930852, 3.9427960, -12.7138624, 12.1442041
1: -5.8131151, 5.5860500, -5.0866971, 4.8952274, -10.7083426, 10.6727467
2: -7.5656533, 5.6316767, -6.5687122, 4.9571090, -12.5227623, 12.2003889
3: -8.0398540, 4.9661331, -6.9561987, 4.3684607, -12.4083147, 11.9223318
4: -8.7556324, 6.9592280, -7.6106443, 6.0861058, -14.8417377, 14.5698719
5: -7.1914692, 5.0650182, -6.2800198, 4.4643998, -11.6558685, 11.3450375
6: -6.6938596, 6.7939858, -5.8260970, 5.9469571, -12.6408167, 12.6200829
7: -7.4068289, 6.7790213, -6.4561410, 5.9536462, -13.3604755, 13.2351627
8: -9.4592066, 5.7437124, -8.2420750, 5.0437174, -14.5029240, 13.9857874
9: -6.5829935, 6.7539654, -5.7539897, 5.9019547, -12.4849482, 12.5079556

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8416564, upper bound: 10.8417081
time: 2.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418486
time: 2.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418501
time: 2.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.6931620, 3.9428768, -8.7710667, 4.4511189, -12.1442814, 12.7139435
1: -5.0867200, 4.8952451, -5.8131151, 5.5860500, -10.6727695, 10.7083607
2: -6.5687504, 4.9571500, -7.5656533, 5.6316767, -12.2004271, 12.5228033
3: -6.9562230, 4.3684926, -8.0398540, 4.9661331, -11.9223557, 12.4083462
4: -7.6106682, 6.0861301, -8.7556324, 6.9592280, -14.5698967, 14.8417625
5: -6.2800751, 4.4644489, -7.1914692, 5.0650182, -11.3450928, 11.6559181
6: -5.8261471, 5.9469910, -6.6938596, 6.7939858, -12.6201324, 12.6408501
7: -6.4561977, 5.9537015, -7.4068289, 6.7790213, -13.2352190, 13.3605309
8: -8.2421150, 5.0437574, -9.4592066, 5.7437124, -13.9858274, 14.5029640
9: -5.7540236, 5.9019938, -6.5829935, 6.7539654, -12.5079889, 12.4849873

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417070, upper bound: 10.8417066
time: 3.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418316, upper bound: 10.8418120
time: 3.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418316, upper bound: 10.8418127
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.8629389, 3.0545459, -7.7609200, 3.9201691, -9.7831078, 10.8154659
1: -3.8907924, 3.7586288, -5.1652188, 4.9742346, -8.8650265, 8.9238472
2: -4.9126577, 3.8263013, -6.6658468, 5.0065084, -9.9191666, 10.4921484
3: -5.2317338, 3.3731439, -7.0862846, 4.4198561, -9.6515903, 10.4594288
4: -5.7443252, 4.6418915, -7.7552509, 6.1850510, -11.9293766, 12.3971424
5: -4.7605915, 3.4332638, -6.3697948, 4.4747143, -9.2353058, 9.8030586
6: -4.3452797, 4.5257974, -5.8586807, 6.0046082, -10.3498878, 10.3844776
7: -4.8717022, 4.5811138, -6.5311246, 6.0210676, -10.8927698, 11.1122379
8: -6.2311954, 3.8681710, -8.3739367, 5.0874991, -11.3186951, 12.2421074
9: -4.3874497, 4.5080042, -5.8456268, 5.9873142, -10.3747635, 10.3536310

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418494, upper bound: 10.8418180
time: 3.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418463, upper bound: 10.8418188
time: 3.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.4367032, 3.3333251, -7.8720102, 3.9749968, -10.4117002, 11.2053356
1: -4.2693806, 4.1175151, -5.2393742, 5.0443249, -9.3137054, 9.3568897
2: -5.4385881, 4.1840410, -6.7682114, 5.0762949, -10.5148830, 10.9522524
3: -5.7669821, 3.6868169, -7.1966228, 4.4812117, -10.2481937, 10.8834400
4: -6.3341417, 5.1012487, -7.8711338, 6.2744088, -12.6085510, 12.9723825
5: -5.2444081, 3.7528450, -6.4644246, 4.5368576, -9.7812653, 10.2172699
6: -4.8062801, 4.9734530, -5.9474192, 6.0918155, -10.8980961, 10.9208717
7: -5.3754444, 5.0135107, -6.6293244, 6.1063948, -11.4818392, 11.6428356
8: -6.8695068, 4.2351508, -8.4987736, 5.1592588, -12.0287657, 12.7339249
9: -4.8222027, 4.9465990, -5.9307365, 6.0748653, -10.8970680, 10.8773355

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418488, upper bound: 10.8418177
time: 3.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418459, upper bound: 10.8418187
time: 3.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.4719820, 3.3499537, -9.1093225, 4.5797076, -11.0516891, 12.4592762
1: -4.2972336, 4.1457505, -6.0533729, 5.8115520, -10.1087856, 10.1991234
2: -5.4767232, 4.2087107, -7.8877387, 5.8447852, -11.3215084, 12.0964489
3: -5.8084311, 3.7097430, -8.4040852, 5.1549087, -10.9633398, 12.1138287
4: -6.3828230, 5.1340618, -9.1370716, 7.2508297, -13.6336527, 14.2711334
5: -5.2791357, 3.7697303, -7.4905796, 5.2391715, -10.5183067, 11.2603102
6: -4.8342590, 5.0014062, -6.9476757, 7.0581341, -11.8923931, 11.9490814
7: -5.4140658, 5.0452414, -7.7037601, 7.0385532, -12.4526196, 12.7490015
8: -6.9161587, 4.2584696, -9.8604088, 5.9603915, -12.8765507, 14.1188784
9: -4.8564644, 4.9799275, -6.8548994, 7.0309110, -11.8873749, 11.8348274

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418731, upper bound: 10.8418482
time: 2.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418729, upper bound: 10.8418461
time: 3.27 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.9624119, 3.5865674, -7.7609200, 3.9201691, -10.8825808, 11.3474874
1: -4.6018915, 4.4345980, -5.1652188, 4.9742346, -9.5761261, 9.5998173
2: -5.8984613, 4.5002947, -6.6658468, 5.0065084, -10.9049702, 11.1661415
3: -6.2346840, 3.9667058, -7.0862846, 4.4198561, -10.6545401, 11.0529900
4: -6.8513365, 5.5012336, -7.7552509, 6.1850510, -13.0363874, 13.2564850
5: -5.6639371, 4.0537653, -6.3697948, 4.4747143, -10.1386509, 10.4235601
6: -5.2382298, 5.3745875, -5.8586807, 6.0046082, -11.2428379, 11.2332687
7: -5.8129950, 5.3952818, -6.5311246, 6.0210676, -11.8340626, 11.9264069
8: -7.4270153, 4.5703878, -8.3739367, 5.0874991, -12.5145149, 12.9443245
9: -5.1979680, 5.3300619, -5.8456268, 5.9873142, -11.1852818, 11.1756887

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 113

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418183
time: 3.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418458, upper bound: 10.8418172
time: 2.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.4975357, 3.8520713, -7.8720102, 3.9749968, -11.4725323, 11.7240810
1: -4.9517870, 4.7649336, -5.2393742, 5.0443249, -9.9961119, 10.0043077
2: -6.3835559, 4.8333735, -6.7682114, 5.0762949, -11.4598503, 11.6015854
3: -6.7537041, 4.2575140, -7.1966228, 4.4812117, -11.2349157, 11.4541368
4: -7.3932443, 5.9240074, -7.8711338, 6.2744088, -13.6676531, 13.7951412
5: -6.1101584, 4.3585858, -6.4644246, 4.5368576, -10.6470165, 10.8230104
6: -5.6704216, 5.7940845, -5.9474192, 6.0918155, -11.7622375, 11.7415037
7: -6.2784386, 5.8002563, -6.6293244, 6.1063948, -12.3848333, 12.4295807
8: -8.0158615, 4.9167080, -8.4987736, 5.1592588, -13.1751204, 13.4154816
9: -5.5974998, 5.7431765, -5.9307365, 6.0748653, -11.6723652, 11.6739130

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418478, upper bound: 10.8418187
time: 3.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418461, upper bound: 10.8418170
time: 3.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1753426, 4.1578913, -8.8251686, 4.4374833, -12.6128254, 12.9830599
1: -5.4232693, 5.2176561, -5.8674273, 5.6387110, -11.0619802, 11.0850830
2: -7.0285602, 5.2623281, -7.6311131, 5.6672015, -12.6957617, 12.8934412
3: -7.4615574, 4.6421537, -8.1280937, 5.0017595, -12.4633169, 12.7702475
4: -8.1493320, 6.4913750, -8.8525677, 7.0266743, -15.1760063, 15.3439426
5: -6.6995029, 4.7251897, -7.2538118, 5.0767117, -11.7762146, 11.9790020
6: -6.2095594, 6.3282332, -6.7196293, 6.8341055, -13.0436649, 13.0478630
7: -6.8913727, 6.3298273, -7.4578910, 6.8254862, -13.7168589, 13.7877178
8: -8.8068333, 5.3575735, -9.5484190, 5.7764950, -14.5833282, 14.9059925
9: -6.1401882, 6.2953801, -6.6444292, 6.8121629, -12.9523506, 12.9398098

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417560, upper bound: 10.8417681
time: 2.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418411, upper bound: 10.8418124
time: 6.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418400, upper bound: 10.8418388
time: 4.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.5572586, 3.8791888, -9.1093225, 4.5797076, -12.1369667, 12.9885111
1: -4.9953742, 4.8081083, -6.0533729, 5.8115520, -10.8069267, 10.8614807
2: -6.4433255, 4.8723674, -7.8877387, 5.8447852, -12.2881107, 12.7601061
3: -6.8195372, 4.2932954, -8.4040852, 5.1549087, -11.9744453, 12.6973801
4: -7.4664049, 5.9759054, -9.1370716, 7.2508297, -14.7172346, 15.1129770
5: -6.1644917, 4.3893023, -7.4905796, 5.2391715, -11.4036636, 11.8798819
6: -5.7175026, 5.8408585, -6.9476757, 7.0581341, -12.7756367, 12.7885342
7: -6.3369069, 5.8498435, -7.7037601, 7.0385532, -13.3754597, 13.5536041
8: -8.0885534, 4.9559178, -9.8604088, 5.9603915, -14.0489445, 14.8163261
9: -5.6494303, 5.7947531, -6.8548994, 7.0309110, -12.6803417, 12.6496525

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 70

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418125
time: 2.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418479
time: 3.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.9208469, 4.4868975, -5.5604835, 2.9136519, -11.8344994, 10.0473804
1: -5.9277759, 5.6934175, -3.6883759, 3.5663452, -9.4941216, 9.3817940
2: -7.7150965, 5.7263079, -4.6331820, 3.6394210, -11.3545170, 10.3594894
3: -8.2174015, 5.0513983, -4.9431987, 3.2068086, -11.4242096, 9.9945965
4: -8.9413919, 7.1000547, -5.4262838, 4.3964443, -13.3378363, 12.5263386
5: -7.3314867, 5.1333966, -4.5021811, 3.2688141, -10.6003008, 9.6355782
6: -6.7965636, 6.9101586, -4.1061349, 4.2902169, -11.0867805, 11.0162935
7: -7.5374422, 6.8948312, -4.6076269, 4.3525953, -11.8900375, 11.5024586
8: -9.6502542, 5.8382874, -5.8892493, 3.6758657, -13.3261204, 11.7275372
9: -6.7114735, 6.8832517, -4.1558294, 4.2745419, -10.9860153, 11.0390816

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418315
time: 2.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418465
time: 2.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.0384598, 4.5449328, -6.1134653, 3.1802311, -12.2186909, 10.6583977
1: -6.0060692, 5.7670245, -4.0579419, 3.9182158, -9.9242849, 9.8249664
2: -7.8227282, 5.8002195, -5.1457872, 3.9846997, -11.8074284, 10.9460068
3: -8.3337669, 5.1159482, -5.4680281, 3.5121632, -11.8459301, 10.5839767
4: -9.0633039, 7.1940422, -6.0069256, 4.8438358, -13.9071398, 13.2009678
5: -7.4307408, 5.1994286, -4.9739480, 3.5728743, -11.0036154, 10.1733761
6: -6.8908195, 7.0024395, -4.5484085, 4.7223129, -11.6131325, 11.5508480
7: -7.6411672, 6.9844575, -5.0985341, 4.7739000, -12.4150677, 12.0829916
8: -9.7812653, 5.9144554, -6.5127282, 4.0294580, -13.8107233, 12.4271832
9: -6.8008847, 6.9753227, -4.5813155, 4.7029467, -11.5038319, 11.5566387

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 113

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418123, upper bound: 10.8418310
time: 2.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418123, upper bound: 10.8418448
time: 2.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8476982, 4.4486094, -7.5353460, 3.8715367, -12.7192345, 11.9839554
1: -5.8834343, 5.6533856, -4.9816504, 4.7951336, -10.6785679, 10.6350365
2: -7.6532927, 5.6822624, -6.4250069, 4.8600698, -12.5133629, 12.1072693
3: -8.1522961, 5.0144453, -6.7991381, 4.2821770, -12.4344730, 11.8135834
4: -8.8770294, 7.0462732, -7.4452305, 5.9592419, -14.8362713, 14.4915037
5: -7.2742939, 5.0891600, -6.1479549, 4.3769145, -11.6512089, 11.2371149
6: -6.7372761, 6.8530869, -5.7000742, 5.8246045, -12.5618801, 12.5531616
7: -7.4789629, 6.8433170, -6.3207989, 5.8349400, -13.3139029, 13.1641159
8: -9.5756159, 5.7913136, -8.0657997, 4.9419928, -14.5176086, 13.8571129
9: -6.6627541, 6.8310509, -5.6347437, 5.7792683, -12.4420223, 12.4657946

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 70

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417844, upper bound: 10.8418009
time: 2.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417850, upper bound: 10.8418168
time: 2.13 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.16 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418395
IS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418406
IS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418486
IS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418109, upper bound: 10.8418501
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418316, upper bound: 10.8418120
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418316, upper bound: 10.8418127
IS_A1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418494, upper bound: 10.8418180
IS_A1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418463, upper bound: 10.8418188
IS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418488, upper bound: 10.8418177
IS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418459, upper bound: 10.8418187
IS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418731, upper bound: 10.8418482
IS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418729, upper bound: 10.8418461
IS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418183
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418458, upper bound: 10.8418172
IS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418478, upper bound: 10.8418187
IS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418461, upper bound: 10.8418170
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418411, upper bound: 10.8418124
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418400, upper bound: 10.8418388
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418125
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418479
IS_A2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418315
IS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418125, upper bound: 10.8418465
IS_A2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418123, upper bound: 10.8418310
IS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8418123, upper bound: 10.8418448
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8417844, upper bound: 10.8418009
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.16
Output dim: 0, lower bound: -10.8417850, upper bound: 10.8418168

## BFS IS instance: IS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -10.8517380, 5.3948221, -8.3535156, 4.2414622, -15.0932007, 13.7483377
1: -7.2513132, 6.9528608, -5.5427141, 5.3322773, -12.5835905, 12.4955750
2: -9.5307407, 6.9504719, -7.1935520, 5.3739529, -14.9046936, 14.1440239
3: -10.2015448, 6.1356497, -7.6416631, 4.7404404, -14.9419851, 13.7773132
4: -11.0363302, 8.6953363, -8.3381834, 6.6367073, -17.6730385, 17.0335197
5: -8.9931183, 6.1996779, -6.8516784, 4.8238721, -13.8169899, 13.0513563
6: -8.3469524, 8.4441128, -6.3519821, 6.4674263, -14.8143787, 14.7960949
7: -9.2685089, 8.3902550, -7.0485158, 6.4663334, -15.7348423, 15.4387703
8: -11.8722782, 7.0981383, -9.0084543, 5.4729772, -17.3452549, 16.1065922
9: -8.2277184, 8.4364910, -6.2780080, 6.4362836, -14.6640015, 14.7144985

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 70

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418128, upper bound: 10.8418378
time: 2.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418130, upper bound: 10.8418408
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3541031, 4.2415218, -7.6930852, 3.9427960, -12.2968988, 11.9346066
1: -5.5442796, 5.3322868, -5.0866971, 4.8952274, -10.4395065, 10.4189835
2: -7.1940002, 5.3745918, -6.5687122, 4.9571090, -12.1511097, 11.9433041
3: -7.6428828, 4.7412558, -6.9561987, 4.3684607, -12.0113430, 11.6974545
4: -8.3398428, 6.6369152, -7.6106443, 6.0861058, -14.4259491, 14.2475595
5: -6.8517056, 4.8241949, -6.2800198, 4.4643998, -11.3161049, 11.1042147
6: -6.3526058, 6.4684858, -5.8260970, 5.9469571, -12.2995625, 12.2945824
7: -7.0491538, 6.4666395, -6.4561410, 5.9536462, -13.0028000, 12.9227810
8: -9.0096455, 5.4731207, -8.2420750, 5.0437174, -14.0533628, 13.7151957
9: -6.2781296, 6.4372911, -5.7539897, 5.9019547, -12.1800842, 12.1912804

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417067, upper bound: 10.8417084
time: 3.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418113, upper bound: 10.8418483
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418113, upper bound: 10.8418490
time: 3.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -10.8517380, 5.3948221, -7.6930852, 3.9427960, -14.7945337, 13.0879078
1: -7.2513132, 6.9528608, -5.0866971, 4.8952274, -12.1465406, 12.0395584
2: -9.5307407, 6.9504719, -6.5687122, 4.9571090, -14.4878502, 13.5191841
3: -10.2015448, 6.1356497, -6.9561987, 4.3684607, -14.5700054, 13.0918484
4: -11.0363302, 8.6953363, -7.6106443, 6.0861058, -17.1224365, 16.3059807
5: -8.9931183, 6.1996779, -6.2800198, 4.4643998, -13.4575176, 12.4796982
6: -8.3469524, 8.4441128, -5.8260970, 5.9469571, -14.2939091, 14.2702103
7: -9.2685089, 8.3902550, -6.4561410, 5.9536462, -15.2221546, 14.8463955
8: -11.8722782, 7.0981383, -8.2420750, 5.0437174, -16.9159966, 15.3402138
9: -8.2277184, 8.4364910, -5.7539897, 5.9019547, -14.1296730, 14.1904812

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 70

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418113, upper bound: 10.8418484
time: 3.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418113, upper bound: 10.8418503
time: 3.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4.3974237, 2.2795186, -7.5726843, 3.8198528, -8.2172766, 9.8522034
1: -2.9251580, 2.8326793, -5.0397205, 4.8557577, -7.7809157, 7.8723998
2: -3.5503275, 2.8995328, -6.4906797, 4.8860602, -8.4363880, 9.3902130
3: -3.8535514, 2.5595558, -6.8998585, 4.3152800, -8.1688309, 9.4594145
4: -4.2226267, 3.4762244, -7.5602412, 6.0337400, -10.2563667, 11.0364656
5: -3.5031080, 2.6147432, -6.2081618, 4.3675027, -7.8706107, 8.8229046
6: -3.1612709, 3.3711734, -5.7061348, 5.8547592, -9.0160303, 9.0773087
7: -3.5390978, 3.4525700, -6.3601322, 5.8747153, -9.4138126, 9.8127022
8: -4.5788631, 2.9316380, -8.1618938, 4.9650097, -9.5438728, 11.0935316
9: -3.2561593, 3.3674440, -5.7002463, 5.8377914, -9.0939503, 9.0676899

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_A1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418465, upper bound: 10.8418180
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418465, upper bound: 10.8418177
time: 2.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -4.7180710, 2.4638691, -7.2006574, 3.6308224, -8.3488932, 9.6645260
1: -3.1307101, 3.0316763, -4.7906327, 4.6202917, -7.7510018, 7.8223090
2: -3.8476157, 3.1057663, -6.1449509, 4.6499705, -8.4975863, 9.2507172
3: -4.1470146, 2.7370877, -6.5283060, 4.1091542, -8.2561684, 9.2653942
4: -4.5468102, 3.7246590, -7.1719995, 5.7326298, -10.2794399, 10.8966579
5: -3.7782927, 2.7966735, -5.8892126, 4.1576743, -7.9359670, 8.6858864
6: -3.4211488, 3.6249404, -5.4065914, 5.5597391, -8.9808884, 9.0315323
7: -3.8398995, 3.7041175, -6.0267868, 5.5877719, -9.4276714, 9.7309046
8: -4.9339414, 3.1365838, -7.7412429, 4.7234035, -9.6573448, 10.8778267
9: -3.5041432, 3.6167092, -5.4132199, 5.5427885, -9.0469322, 9.0299292

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_A1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=13.22586727142334
rel_dist={0: [-10.84189645044881, 10.84189726697796]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1353.63 seconds
