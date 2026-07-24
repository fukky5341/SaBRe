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
execution time: IAR + LP analysis = 1.05 + 9.83 = 10.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -202.6091898, upper bound: 202.6091898


# Binary Search by BASE starts (time budget: 2689.12 seconds, max iter: 100)

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
Binary search time: 37.49 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2651.63 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5741901, upper bound: 202.5684197
time: 7.86 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5559891, upper bound: 202.5559891
time: 5.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.42
Output dim: 1, lower bound: -202.5741901, upper bound: 202.5684197
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.42
Output dim: 1, lower bound: -202.5559891, upper bound: 202.5559891

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -129.5555573, 103.5026779, -130.5882568, 104.3158493, -233.8713989, 234.0909424
1: -110.1596298, 91.7861023, -111.0153198, 92.5080414, -202.6676636, 202.8014221
2: -143.4482269, 93.1944199, -144.5845337, 93.9274826, -237.3757019, 237.7789459
3: -151.3298798, 80.6287231, -152.5511169, 81.2579956, -232.5878601, 233.1798248
4: -139.4388275, 107.4287872, -140.5548248, 108.2794342, -247.7182617, 247.9836121
5: -124.6241226, 97.0947723, -125.6117477, 97.8683395, -222.4924622, 222.7065125
6: -119.7312012, 115.7826080, -120.6793823, 116.6907272, -236.4219360, 236.4619751
7: -130.0381012, 109.5673599, -131.0779724, 110.4361649, -240.4742432, 240.6453247
8: -158.2102203, 109.4822311, -159.4465179, 110.3260040, -268.5361938, 268.9287109
9: -118.7252274, 117.8584976, -119.6686935, 118.7806778, -237.5059052, 237.5271606

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5423220, upper bound: 202.5384559
time: 8.37 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5129275, upper bound: 202.5155008
time: 7.97 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5735370, upper bound: 202.5676991
time: 7.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.5170898, 100.3681488, -129.5802917, 103.5229187, -229.0400085, 229.9484406
1: -107.0420685, 89.0140533, -110.1828995, 91.8047791, -198.8468323, 199.1969452
2: -139.0532227, 90.3464966, -143.4771729, 93.2133636, -232.2665405, 233.8236694
3: -146.3701935, 78.2022858, -151.3567810, 80.6423264, -227.0125122, 229.5590668
4: -135.0138855, 104.1010361, -139.4685059, 107.4520264, -242.4658966, 243.5695190
5: -120.8127518, 94.0281754, -124.6453476, 97.1162109, -217.9289246, 218.6735077
6: -116.0574951, 112.3338470, -119.7536011, 115.8065033, -231.8639984, 232.0874176
7: -125.9195633, 106.1953506, -130.0666656, 109.5913162, -235.5108795, 236.2620087
8: -153.5792236, 106.2095566, -158.2426605, 109.5036240, -263.0828247, 264.4521484
9: -115.0688782, 114.3207016, -118.7528305, 117.8851318, -232.9540100, 233.0735321

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5092292, upper bound: 202.5128199
time: 7.22 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 5.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.54 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 1, lower bound: -202.5129275, upper bound: 202.5155008
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 1, lower bound: -202.5735370, upper bound: 202.5676991
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 1, lower bound: -202.5092292, upper bound: 202.5128199
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.54
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -121.2584915, 96.9511490, -129.3389740, 103.3275375, -224.5860138, 226.2901154
1: -103.2872314, 86.0529861, -109.9729309, 91.6383896, -194.9256134, 196.0258942
2: -134.4150391, 87.1443481, -143.2169647, 93.0265274, -227.4415588, 230.3613129
3: -141.8006744, 75.4621811, -151.1020966, 80.4852905, -222.2859650, 226.5642700
4: -130.7071991, 100.7575912, -139.2247620, 107.2630386, -237.9702301, 239.9823608
5: -116.6296616, 90.9831390, -124.4093018, 96.9436188, -213.5732727, 215.3924408
6: -112.1325912, 108.5741577, -119.5321198, 115.5943146, -227.7268982, 228.1062775
7: -121.9180298, 102.7201080, -129.8419952, 109.3963318, -231.3143311, 232.5620728
8: -148.2859802, 102.3969955, -157.9449463, 109.2746582, -257.5606384, 260.3419189
9: -111.4574356, 110.5542068, -118.5514755, 117.6693573, -229.1267700, 229.1056519

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 23

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4820881, upper bound: 202.4846206
time: 8.07 seconds

## Relational analysis of IS_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4898089, upper bound: 202.4932887
time: 7.36 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5102521, upper bound: 202.5130675
time: 7.88 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -128.5523529, 102.7080536, -130.5882568, 104.3158493, -232.8681946, 233.2963104
1: -109.3197327, 91.0861740, -111.0153198, 92.5080414, -201.8277740, 202.1014862
2: -142.3439026, 92.4743347, -144.5845337, 93.9274826, -236.2713928, 237.0588684
3: -150.1590881, 80.0108566, -152.5511169, 81.2579956, -231.4170685, 232.5619812
4: -138.3659363, 106.6108246, -140.5548248, 108.2794342, -246.6453705, 247.1656342
5: -123.6577225, 96.3522949, -125.6117477, 97.8683395, -221.5260620, 221.9640350
6: -118.8093338, 114.8993835, -120.6793823, 116.6907272, -235.5000610, 235.5787659
7: -129.0399780, 108.7297974, -131.0779724, 110.4361649, -239.4761200, 239.8077698
8: -157.0040588, 108.6439514, -159.4465179, 110.3260040, -267.3300171, 268.0904541
9: -117.8264618, 116.9623718, -119.6686935, 118.7806778, -236.6071472, 236.6310730

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5414854, upper bound: 202.5377021
time: 6.75 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5621481, upper bound: 202.5576978
time: 8.50 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5234044, upper bound: 202.5228475
time: 7.55 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5735370, upper bound: 202.5676991
time: 7.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.5170898, 100.3681488, -122.8679428, 98.1971970, -223.7142944, 223.2360840
1: -107.0420685, 89.0140533, -104.5816803, 87.0694733, -194.1115265, 193.5957184
2: -139.0532227, 90.3464966, -136.0710297, 88.3924103, -227.4456177, 226.4175262
3: -146.3701935, 78.2022858, -143.3650513, 76.4781570, -222.8483276, 221.5673370
4: -135.0138855, 104.1010361, -132.2326202, 101.9615784, -236.9754639, 236.3336487
5: -120.8127518, 94.0281754, -118.1490555, 92.0646210, -212.8773651, 212.1772308
6: -116.0574951, 112.3338470, -113.5710983, 109.8652039, -225.9226990, 225.9049377
7: -125.9195633, 106.1953506, -123.3063431, 103.9287796, -229.8483276, 229.5016937
8: -153.5792236, 106.2095566, -150.2139282, 104.0473633, -257.6265869, 256.4234619
9: -115.0688782, 114.3207016, -112.6510468, 111.8445969, -226.9134827, 226.9717407

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 5.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 5.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -124.5834274, 99.6296234, -125.6269684, 100.4216537, -225.0050507, 225.2565765
1: -106.2626190, 88.3553772, -106.9017334, 88.9960861, -195.2586975, 195.2571106
2: -138.0224762, 89.6741791, -139.1017303, 90.3132553, -228.3357239, 228.7759094
3: -145.2582855, 77.6252823, -146.5878906, 78.1592255, -223.4175110, 224.2131653
4: -134.0056152, 103.3355026, -135.1705780, 104.1812057, -238.1868286, 238.5060730
5: -119.9113388, 93.3244858, -120.8034897, 94.0786057, -213.9899445, 214.1279755
6: -115.1974182, 111.5064316, -116.1093521, 112.3068237, -227.5042419, 227.6157837
7: -124.9767914, 105.4064713, -126.0349503, 106.2402649, -231.2170563, 231.4414215
8: -152.4625244, 105.4528275, -153.5945740, 106.3244629, -258.7869873, 259.0473938
9: -114.2173920, 113.4792252, -115.1334076, 114.3177109, -228.5350952, 228.6126099

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 5.06 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
time: 5.11 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 55.32 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.4898089, upper bound: 202.4932887
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.5102521, upper bound: 202.5130675
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.5234044, upper bound: 202.5228475
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.5735370, upper bound: 202.5676991
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 55.32
Output dim: 1, lower bound: -202.4960197, upper bound: 202.4960197

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -121.1019669, 96.8273926, -121.7943192, 97.3499374, -218.4519043, 218.6217041
1: -103.1544342, 85.9426117, -103.5839081, 86.3160553, -189.4704742, 189.5265198
2: -134.2418976, 87.0326309, -134.8705292, 87.6268082, -221.8687134, 221.9031525
3: -141.6168671, 75.3650513, -142.2480774, 75.8257294, -217.4425964, 217.6131287
4: -130.5384979, 100.6281433, -131.0886536, 101.0294952, -231.5679932, 231.7167969
5: -116.4800415, 90.8657837, -117.1727524, 91.2900391, -207.7700653, 208.0385132
6: -111.9890823, 108.4350662, -112.6052933, 108.8915176, -220.8806000, 221.0403595
7: -121.7591324, 102.5873032, -122.2075424, 103.0025787, -224.7617188, 224.7948456
8: -148.0978699, 102.2684631, -148.8392487, 103.0677795, -251.1656342, 251.1077118
9: -111.3135452, 110.4128647, -111.6266251, 110.8763428, -222.1898804, 222.0394897

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4757674, upper bound: 202.4787408
time: 8.28 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4757674, upper bound: 202.4932887
time: 8.18 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -121.2436905, 96.9395142, -124.5075226, 99.5234680, -220.7671509, 221.4470062
1: -103.2747192, 86.0425644, -105.8797226, 88.2365570, -191.5112762, 191.9222870
2: -134.3987274, 87.1338425, -137.8890076, 89.5954666, -223.9942017, 225.0228424
3: -141.7832642, 75.4529572, -145.4150848, 77.4765244, -219.2597504, 220.8680420
4: -130.6912994, 100.7453995, -134.0341187, 103.2792282, -233.9705200, 234.7795105
5: -116.6155243, 90.9720840, -119.7945862, 93.3346863, -209.9502106, 210.7666626
6: -112.1190491, 108.5610428, -115.1094284, 111.3099442, -223.4289856, 223.6704712
7: -121.9030533, 102.7076263, -124.9553452, 105.3215332, -227.2245789, 227.6629639
8: -148.2682800, 102.3848877, -152.1687164, 105.3164520, -253.5847321, 254.5536041
9: -111.4439316, 110.5409012, -114.1417847, 113.3212509, -224.7651825, 224.6826630

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -202.4796299, upper bound: 202.4822979
time: 8.40 seconds

## Relational analysis of IS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4994414, upper bound: 202.5028159
time: 7.81 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5069206, upper bound: 202.5100694
time: 7.48 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -125.3214340, 100.1299973, -129.8467712, 103.7257843, -229.0472107, 229.9767761
1: -106.6139984, 88.8545456, -110.3908539, 91.9891968, -198.6031799, 199.2453766
2: -138.7927856, 90.1553268, -143.7674561, 93.3966980, -232.1894836, 233.9227905
3: -146.3778992, 78.0392456, -151.6802521, 80.8032379, -227.1811371, 229.7194824
4: -134.7587128, 103.9499664, -139.7461395, 107.6702347, -242.4289398, 243.6960907
5: -120.5432053, 93.8951721, -124.8977280, 97.3119431, -217.8551178, 218.7928772
6: -115.7989655, 112.0262451, -119.9934387, 116.0320969, -231.8310394, 232.0196838
7: -125.7162323, 106.0480728, -130.3265381, 109.8141327, -235.5303650, 236.3745880
8: -153.1678925, 105.9054413, -158.5539703, 109.7123337, -262.8802185, 264.4593811
9: -114.8478699, 114.0098953, -118.9894257, 118.1097794, -232.9576416, 232.9993286

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5190451, upper bound: 202.5176214
time: 8.24 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5188957, upper bound: 202.5175761
time: 8.57 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -127.5905914, 101.9440079, -130.5882568, 104.3158493, -231.9064331, 232.5322571
1: -108.5131836, 90.4167328, -111.0153198, 92.5080414, -201.0212250, 201.4320526
2: -141.2881165, 91.7863998, -144.5845337, 93.9274826, -235.2156067, 236.3709259
3: -149.0323486, 79.4222260, -152.5511169, 81.2579956, -230.2903290, 231.9733429
4: -137.3187103, 105.8247375, -140.5548248, 108.2794342, -245.5981445, 246.3795624
5: -122.7316055, 95.6317902, -125.6117477, 97.8683395, -220.5999298, 221.2435303
6: -117.9218140, 114.0480576, -120.6793823, 116.6907272, -234.6125336, 234.7274475
7: -128.0689240, 107.9264603, -131.0779724, 110.4361649, -238.5050507, 239.0044250
8: -155.8498077, 107.8488388, -159.4465179, 110.3260040, -266.1758118, 267.2953491
9: -116.9525986, 116.0969849, -119.6686935, 118.7806778, -235.7332764, 235.7656708

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5414854, upper bound: 202.5377002
time: 7.54 seconds

## Relational analysis of IS_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5621481, upper bound: 202.5576978
time: 7.54 seconds

## Relational analysis of IS_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5638277, upper bound: 202.5590556
time: 7.28 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5291597, upper bound: 202.5211642
time: 7.44 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -119.1207123, 95.2933044, -122.8679428, 98.1971970, -217.3179016, 218.1612091
1: -101.7024689, 84.4994736, -104.5816803, 87.0694733, -188.7719269, 189.0811462
2: -132.0008392, 85.7531891, -136.0710297, 88.3924103, -220.3932495, 221.8242188
3: -138.7515564, 74.2292023, -143.3650513, 76.4781570, -215.2296753, 217.5942535
4: -128.1185913, 98.8681107, -132.2326202, 101.9615784, -230.0801697, 231.1007385
5: -114.6191101, 89.2131500, -118.1490555, 92.0646210, -206.6837311, 207.3621979
6: -110.1612473, 106.6732864, -113.5710983, 109.8652039, -220.0264130, 220.2443848
7: -119.4803238, 100.8000107, -123.3063431, 103.9287796, -223.4090881, 224.1063538
8: -145.9260712, 101.0085983, -150.2139282, 104.0473633, -249.9734192, 251.2225342
9: -109.2540741, 108.5624924, -112.6510468, 111.8445969, -221.0986328, 221.2135315

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5012434, upper bound: 202.5037993
time: 5.70 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5062699, upper bound: 202.5099205
time: 5.55 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -120.8375931, 96.6938324, -122.8679428, 98.1971970, -219.0347595, 219.5617676
1: -103.1521378, 85.6930542, -104.5816803, 87.0694733, -190.2216034, 190.2747345
2: -133.8718872, 86.9236679, -136.0710297, 88.3924103, -222.2642975, 222.9946899
3: -140.7424774, 75.2777405, -143.3650513, 76.4781570, -217.2206116, 218.6427917
4: -129.9369659, 100.2361221, -132.2326202, 101.9615784, -231.8985443, 232.4687500
5: -116.2725449, 90.4474182, -118.1490555, 92.0646210, -208.3371582, 208.5964661
6: -111.7457657, 108.1897964, -113.5710983, 109.8652039, -221.6109619, 221.7608948
7: -121.1550674, 102.2235947, -123.3063431, 103.9287796, -225.0838165, 225.5299377
8: -148.0612183, 102.4433517, -150.2139282, 104.0473633, -252.1085663, 252.6572876
9: -110.7916107, 110.1029129, -112.6510468, 111.8445969, -222.6362000, 222.7539215

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5012434, upper bound: 202.5037993
time: 7.29 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5062699, upper bound: 202.5099205
time: 5.82 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -119.1207123, 95.2933044, -125.6269684, 100.4216537, -219.5423279, 220.9202423
1: -101.7024689, 84.4994736, -106.9017334, 88.9960861, -190.6985474, 191.4011993
2: -132.0008392, 85.7531891, -139.1017303, 90.3132553, -222.3140869, 224.8549194
3: -138.7515564, 74.2292023, -146.5878906, 78.1592255, -216.9107819, 220.8170929
4: -128.1185913, 98.8681107, -135.1705780, 104.1812057, -232.2998047, 234.0386963
5: -114.6191101, 89.2131500, -120.8034897, 94.0786057, -208.6977234, 210.0166321
6: -110.1612473, 106.6732864, -116.1093521, 112.3068237, -222.4680634, 222.7826385
7: -119.4803238, 100.8000107, -126.0349503, 106.2402649, -225.7205811, 226.8349457
8: -145.9260712, 101.0085983, -153.5945740, 106.3244629, -252.2505188, 254.6031799
9: -109.2540741, 108.5624924, -115.1334076, 114.3177109, -223.5717773, 223.6958923

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=203.5233612060547
rel_dist={1: [-202.60902678108835, 202.60902678108835]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5653810, upper bound: 202.5624047
time: 7.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5556257, upper bound: 202.5556257
time: 6.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.13
Output dim: 1, lower bound: -202.5653810, upper bound: 202.5624047
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.13
Output dim: 1, lower bound: -202.5556257, upper bound: 202.5556257

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -129.5555573, 103.5026779, -130.5882568, 104.3158493, -233.8713989, 234.0909424
1: -110.1596298, 91.7861023, -111.0153198, 92.5080414, -202.6676636, 202.8014221
2: -143.4482269, 93.1944199, -144.5845337, 93.9274826, -237.3757019, 237.7789459
3: -151.3298798, 80.6287231, -152.5511169, 81.2579956, -232.5878601, 233.1798248
4: -139.4388275, 107.4287872, -140.5548248, 108.2794342, -247.7182617, 247.9836121
5: -124.6241226, 97.0947723, -125.6117477, 97.8683395, -222.4924622, 222.7065125
6: -119.7312012, 115.7826080, -120.6793823, 116.6907272, -236.4219360, 236.4619751
7: -130.0381012, 109.5673599, -131.0779724, 110.4361649, -240.4742432, 240.6453247
8: -158.2102203, 109.4822311, -159.4465179, 110.3260040, -268.5361938, 268.9287109
9: -118.7252274, 117.8584976, -119.6686935, 118.7806778, -237.5059052, 237.5271606

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5466341, upper bound: 202.5449670
time: 8.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
time: 8.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.5170898, 100.3681488, -127.8400116, 102.1539307, -227.6710052, 228.2081604
1: -107.0420685, 89.0140533, -108.7457428, 90.5906143, -197.6326599, 197.7597809
2: -139.0532227, 90.3464966, -141.5655670, 91.9802780, -231.0334778, 231.9120636
3: -146.3701935, 78.2022858, -149.2944336, 79.5793457, -225.9495239, 227.4967194
4: -135.0138855, 104.1010361, -137.5931396, 106.0234985, -241.0373840, 241.6941681
5: -120.8127518, 94.0281754, -122.9767609, 95.8180313, -216.6307678, 217.0049133
6: -116.0574951, 112.3338470, -118.1550751, 114.2799683, -230.3374634, 230.4889221
7: -125.9195633, 106.1953506, -128.3204651, 108.1329193, -234.0524750, 234.5158081
8: -153.5792236, 106.2095566, -156.1639404, 108.0838852, -261.6630554, 262.3734741
9: -115.0688782, 114.3207016, -117.1717758, 116.3390045, -231.4078827, 231.4924774

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5025982, upper bound: 202.5043019
time: 6.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
time: 5.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.26
Output dim: 1, lower bound: -202.5466341, upper bound: 202.5449670
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.26
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.26
Output dim: 1, lower bound: -202.5025982, upper bound: 202.5043019
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.26
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -127.6809464, 102.0155640, -123.8538513, 98.9724960, -226.6534424, 225.8694000
1: -108.5956573, 90.4640045, -105.3952637, 87.7574692, -196.3531189, 195.8592682
2: -141.3795776, 91.8484268, -137.1537170, 89.0909424, -230.4705200, 229.0021362
3: -149.0974426, 79.4667435, -144.5335236, 77.0802994, -226.1777344, 224.0002747
4: -137.4180908, 105.8956757, -133.2951508, 102.7706909, -240.1887665, 239.1907959
5: -122.8101044, 95.6843643, -119.0941010, 92.8000793, -215.6101837, 214.7784729
6: -118.0051727, 114.1232300, -114.4768295, 110.7297516, -228.7348938, 228.6000366
7: -128.1505280, 107.9858322, -124.2956467, 104.7549286, -232.9054413, 232.2814636
8: -155.9680023, 107.9590530, -151.3909607, 104.8513794, -260.8193970, 259.3499756
9: -117.0215073, 116.1718216, -113.5468292, 112.7201843, -229.7416687, 229.7186432

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
time: 8.10 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
time: 8.03 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -126.0503159, 100.7290268, -126.6404724, 101.2187271, -227.2690430, 227.3695068
1: -107.2296524, 89.3132324, -107.7386322, 89.7036743, -196.9333191, 197.0518646
2: -139.5784302, 90.6704102, -140.2154541, 91.0321960, -230.6106110, 230.8858643
3: -147.1593475, 78.4596863, -147.7894592, 78.7781143, -225.9374390, 226.2491455
4: -135.6521149, 104.5517120, -136.2635803, 105.0130539, -240.6651611, 240.8152771
5: -121.2394562, 94.4544601, -121.7752228, 94.8353653, -216.0748291, 216.2296753
6: -116.5010910, 112.6743698, -117.0404968, 113.1959991, -229.6970825, 229.7148132
7: -126.4984741, 106.6078720, -127.0528412, 107.0904388, -233.5889130, 233.6607056
8: -154.0137482, 106.6347580, -154.8047180, 107.1508331, -261.1645813, 261.4394836
9: -115.5255203, 114.6980515, -116.0544891, 115.2182541, -230.7437592, 230.7525177

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
time: 8.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
time: 7.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -123.7420273, 98.9603424, -121.1668930, 96.8595200, -220.6015472, 220.1272278
1: -105.5609055, 87.7613297, -103.1776733, 85.8825302, -191.4434052, 190.9389954
2: -137.0959778, 89.0717773, -134.2033844, 87.1872864, -224.2832642, 223.2751617
3: -144.2558594, 77.1000595, -141.3488007, 75.4390488, -219.6949158, 218.4488373
4: -133.1004486, 102.6491623, -130.3995972, 100.5653076, -233.6657562, 233.0487366
5: -119.0944901, 92.6920013, -116.5183258, 90.7959366, -209.8904266, 209.2103271
6: -114.4218140, 110.7630005, -112.0081406, 108.3737640, -222.7955780, 222.7711487
7: -124.1323318, 104.6979294, -121.5995712, 102.5036697, -226.6360016, 226.2975006
8: -151.4559174, 104.7667847, -148.1829987, 102.6597977, -254.1157227, 252.9497528
9: -113.4551697, 112.7230759, -111.1055984, 110.3339233, -223.7890930, 223.8286591

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
time: 5.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
time: 5.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -121.9252014, 97.5268860, -123.8764267, 99.0449448, -220.9701538, 221.4033051
1: -104.0433044, 86.4800568, -105.4562454, 87.7740097, -191.8173218, 191.9362946
2: -135.0882263, 87.7594452, -137.1783447, 89.0717697, -224.1600037, 224.9377899
3: -142.0933533, 75.9822464, -144.5126648, 77.0901718, -219.1835022, 220.4949036
4: -131.1350861, 101.1555023, -133.2833099, 102.7444916, -233.8795776, 234.4388123
5: -117.3447495, 91.3207550, -119.1249161, 92.7720184, -210.1167603, 210.4456787
6: -112.7483444, 109.1507797, -114.5011444, 110.7712097, -223.5195618, 223.6519165
7: -122.2931976, 103.1603394, -124.2771378, 104.7718658, -227.0650482, 227.4374695
8: -149.2828827, 103.2971725, -151.5046234, 104.8970718, -254.1799469, 254.8017883
9: -111.7926407, 111.0838013, -113.5427246, 112.7625427, -224.5551758, 224.6265259

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
time: 6.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
time: 5.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.17 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.5133731, upper bound: 202.5096892
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.17
Output dim: 1, lower bound: -202.4955996, upper bound: 202.4955996

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -122.8092422, 98.1501846, -123.8538513, 98.9724960, -221.7817383, 222.0040283
1: -104.5302505, 87.0270538, -105.3952637, 87.7574692, -192.2877197, 192.4223175
2: -136.0042725, 88.3492203, -137.1537170, 89.0909424, -225.0952148, 225.5029297
3: -143.2971344, 76.4444122, -144.5335236, 77.0802994, -220.3774414, 220.9779053
4: -132.1659698, 101.9106903, -133.2951508, 102.7706909, -234.9366608, 235.2058411
5: -118.0950165, 92.0176163, -119.0941010, 92.8000793, -210.8950958, 211.1117096
6: -113.5173874, 109.8113861, -114.4768295, 110.7297516, -224.2470856, 224.2882080
7: -123.2436981, 103.8757477, -124.2956467, 104.7549286, -227.9986115, 228.1713867
8: -150.1410675, 103.9992447, -151.3909607, 104.8513794, -254.9924469, 255.3901825
9: -112.5926743, 111.7876053, -113.5468292, 112.7201843, -225.3128662, 225.3344421

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5131877, upper bound: 202.5117028
time: 8.84 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5370183, upper bound: 202.5351653
time: 8.44 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5436238, upper bound: 202.5419516
time: 9.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -125.6385498, 100.4300919, -123.8538513, 98.9724960, -224.6110382, 224.2839355
1: -106.9093170, 89.0033340, -105.3952637, 87.7574692, -194.6667786, 194.3985901
2: -139.1137085, 90.3213806, -137.1537170, 89.0909424, -228.2046509, 227.4750977
3: -146.6041870, 78.1678162, -144.5335236, 77.0802994, -223.6844788, 222.7013397
4: -135.1812897, 104.1876831, -133.2951508, 102.7706909, -237.9519806, 237.4828033
5: -120.8170013, 94.0852127, -119.0941010, 92.8000793, -213.6170807, 213.1793213
6: -116.1203461, 112.3156128, -114.4768295, 110.7297516, -226.8500671, 226.7924500
7: -126.0447998, 106.2477798, -124.2956467, 104.7549286, -230.7997131, 230.5433807
8: -153.6060638, 106.3330536, -151.3909607, 104.8513794, -258.4574585, 257.7239990
9: -115.1392822, 114.3239288, -113.5468292, 112.7201843, -227.8594666, 227.8707581

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5421875, upper bound: 202.5404456
time: 7.74 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5436238, upper bound: 202.5419516
time: 8.97 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -122.8092422, 98.1501846, -126.6404724, 101.2187271, -224.0279694, 224.7906494
1: -104.5302505, 87.0270538, -107.7386322, 89.7036743, -194.2339172, 194.7656860
2: -136.0042725, 88.3492203, -140.2154541, 91.0321960, -227.0364685, 228.5646667
3: -143.2971344, 76.4444122, -147.7894592, 78.7781143, -222.0752106, 224.2338257
4: -132.1659698, 101.9106903, -136.2635803, 105.0130539, -237.1790161, 238.1742706
5: -118.0950165, 92.0176163, -121.7752228, 94.8353653, -212.9303894, 213.7928162
6: -113.5173874, 109.8113861, -117.0404968, 113.1959991, -226.7133789, 226.8518677
7: -123.2436981, 103.8757477, -127.0528412, 107.0904388, -230.3341370, 230.9285889
8: -150.1410675, 103.9992447, -154.8047180, 107.1508331, -257.2919006, 258.8039551
9: -112.5926743, 111.7876053, -116.0544891, 115.2182541, -227.8109131, 227.8421021

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4883969, upper bound: 202.4867441
time: 8.10 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5104859, upper bound: 202.5062849
time: 8.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5105889, upper bound: 202.5064049
time: 8.28 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -125.6385498, 100.4300919, -126.6404724, 101.2187271, -226.8572693, 227.0705566
1: -106.9093170, 89.0033340, -107.7386322, 89.7036743, -196.6129913, 196.7419739
2: -139.1137085, 90.3213806, -140.2154541, 91.0321960, -230.1459045, 230.5368347
3: -146.6041870, 78.1678162, -147.7894592, 78.7781143, -225.3822937, 225.9572754
4: -135.1812897, 104.1876831, -136.2635803, 105.0130539, -240.1943359, 240.4512482
5: -120.8170013, 94.0852127, -121.7752228, 94.8353653, -215.6523743, 215.8604431
6: -116.1203461, 112.3156128, -117.0404968, 113.1959991, -229.3163452, 229.3560944
7: -126.0447998, 106.2477798, -127.0528412, 107.0904388, -233.1352386, 233.3005981
8: -153.6060638, 106.3330536, -154.8047180, 107.1508331, -260.7568970, 261.1377563
9: -115.1392822, 114.3239288, -116.0544891, 115.2182541, -230.3575287, 230.3784180

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4883969, upper bound: 202.4867441
time: 8.06 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5104859, upper bound: 202.5062849
time: 8.85 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5105889, upper bound: 202.5064049
time: 8.53 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -119.1207123, 95.2933044, -121.1668930, 96.8595200, -215.9802246, 216.4601593
1: -101.7024689, 84.4994736, -103.1776733, 85.8825302, -187.5849915, 187.6771545
2: -132.0008392, 85.7531891, -134.2033844, 87.1872864, -219.1881256, 219.9565735
3: -138.7515564, 74.2292023, -141.3488007, 75.4390488, -214.1906128, 215.5780029
4: -128.1185913, 98.8681107, -130.3995972, 100.5653076, -228.6838989, 229.2677002
5: -114.6191101, 89.2131500, -116.5183258, 90.7959366, -205.4150391, 205.7314606
6: -110.1612473, 106.6732864, -112.0081406, 108.3737640, -218.5349884, 218.6814270
7: -119.4803238, 100.8000107, -121.5995712, 102.5036697, -221.9839935, 222.3995819
8: -145.9260712, 101.0085983, -148.1829987, 102.6597977, -248.5858612, 249.1915894
9: -109.2540741, 108.5624924, -111.1055984, 110.3339233, -219.5879974, 219.6680756

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4869330, upper bound: 202.4873800
time: 6.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4995208, upper bound: 202.5013043
time: 6.09 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -120.8375931, 96.6938324, -121.1668930, 96.8595200, -217.6970978, 217.8607178
1: -103.1521378, 85.6930542, -103.1776733, 85.8825302, -189.0346680, 188.8707275
2: -133.8718872, 86.9236679, -134.2033844, 87.1872864, -221.0591736, 221.1270447
3: -140.7424774, 75.2777405, -141.3488007, 75.4390488, -216.1815186, 216.6265411
4: -129.9369659, 100.2361221, -130.3995972, 100.5653076, -230.5022736, 230.6357117
5: -116.2725449, 90.4474182, -116.5183258, 90.7959366, -207.0684814, 206.9657440
6: -111.7457657, 108.1897964, -112.0081406, 108.3737640, -220.1195374, 220.1979218
7: -121.1550674, 102.2235947, -121.5995712, 102.5036697, -223.6587219, 223.8231659
8: -148.0612183, 102.4433517, -148.1829987, 102.6597977, -250.7210083, 250.6263428
9: -110.7916107, 110.1029129, -111.1055984, 110.3339233, -221.1255341, 221.2084503

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4869330, upper bound: 202.4873800
time: 6.84 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4995208, upper bound: 202.5013043
time: 6.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -119.1207123, 95.2933044, -123.8764267, 99.0449448, -218.1656494, 219.1696777
1: -101.7024689, 84.4994736, -105.4562454, 87.7740097, -189.4764709, 189.9557190
2: -132.0008392, 85.7531891, -137.1783447, 89.0717697, -221.0726013, 222.9315338
3: -138.7515564, 74.2292023, -144.5126648, 77.0901718, -215.8416901, 218.7418671
4: -128.1185913, 98.8681107, -133.2833099, 102.7444916, -230.8630829, 232.1514282
5: -114.6191101, 89.2131500, -119.1249161, 92.7720184, -207.3910980, 208.3380280
6: -110.1612473, 106.6732864, -114.5011444, 110.7712097, -220.9324646, 221.1744385
7: -119.4803238, 100.8000107, -124.2771378, 104.7718658, -224.2521820, 225.0771332
8: -145.9260712, 101.0085983, -151.5046234, 104.8970718, -250.8231354, 252.5132141
9: -109.2540741, 108.5624924, -113.5427246, 112.7625427, -222.0165863, 222.1052246

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4918061, upper bound: 202.4918082
time: 6.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4917118, upper bound: 202.4917118
time: 5.92 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -120.8375931, 96.6938324, -123.8764267, 99.0449448, -219.8825378, 220.5702515
1: -103.1521378, 85.6930542, -105.4562454, 87.7740097, -190.9261475, 191.1492920
2: -133.8718872, 86.9236679, -137.1783447, 89.0717697, -222.9436646, 224.1020203
3: -140.7424774, 75.2777405, -144.5126648, 77.0901718, -217.8326263, 219.7904053
4: -129.9369659, 100.2361221, -133.2833099, 102.7444916, -232.6814575, 233.5194397
5: -116.2725449, 90.4474182, -119.1249161, 92.7720184, -209.0445404, 209.5723114
6: -111.7457657, 108.1897964, -114.5011444, 110.7712097, -222.5169678, 222.6909485
7: -121.1550674, 102.2235947, -124.2771378, 104.7718658, -225.9269104, 226.5007324
8: -148.0612183, 102.4433517, -151.5046234, 104.8970718, -252.9582825, 253.9479675
9: -110.7916107, 110.1029129, -113.5427246, 112.7625427, -223.5541534, 223.6456146

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4918061, upper bound: 202.4918082
time: 5.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4917118, upper bound: 202.4917118
time: 5.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 48.12 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5370183, upper bound: 202.5351653
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5436238, upper bound: 202.5419516
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5421875, upper bound: 202.5404456
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5436238, upper bound: 202.5419516
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5104859, upper bound: 202.5062849
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5105889, upper bound: 202.5064049
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5104859, upper bound: 202.5062849
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.5105889, upper bound: 202.5064049
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4869330, upper bound: 202.4873800
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4995208, upper bound: 202.5013043
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4869330, upper bound: 202.4873800
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4995208, upper bound: 202.5013043
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4918061, upper bound: 202.4918082
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4917118, upper bound: 202.4917118
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4918061, upper bound: 202.4918082
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 48.12
Output dim: 1, lower bound: -202.4917118, upper bound: 202.4917118

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -120.8515701, 96.6029053, -116.3794098, 93.0527420, -213.9043121, 212.9823151
1: -102.8694839, 85.6460571, -99.0667648, 82.4841995, -185.3536835, 184.7128296
2: -133.8384552, 86.9515686, -128.8853912, 83.7409744, -217.5794220, 215.8369598
3: -140.9970245, 75.2302780, -135.7595673, 72.4639435, -213.4609375, 210.9898376
4: -130.0547028, 100.2915039, -125.2321701, 96.5972214, -226.6519165, 225.5236816
5: -116.2237778, 90.5494766, -111.9270096, 87.1995468, -203.4233093, 202.4764862
6: -111.7208557, 108.0708466, -107.6119614, 104.0905914, -215.8114471, 215.6828003
7: -121.2554245, 102.2141037, -116.7312241, 98.4193268, -219.6747437, 218.9453278
8: -147.7881165, 102.3932495, -142.3732300, 98.7050323, -246.4931488, 244.7664795
9: -110.7918015, 110.0189819, -106.6846542, 105.9916763, -216.7834778, 216.7036438

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=203.5233612060547
rel_dist={1: [-202.60871310808878, 202.60871310808875]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5382022, upper bound: 202.5369197
time: 9.94 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5335490, upper bound: 202.5335490
time: 6.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.92
Output dim: 1, lower bound: -202.5382022, upper bound: 202.5369197
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.92
Output dim: 1, lower bound: -202.5335490, upper bound: 202.5335490

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -130.1204224, 103.9453583, -130.5882568, 104.3158493, -234.4362640, 234.5336151
1: -110.6282425, 92.1818924, -111.0153198, 92.5080414, -203.1362915, 203.1972046
2: -144.0713348, 93.5925598, -144.5845337, 93.9274826, -237.9988098, 238.1770935
3: -152.0025940, 80.9696350, -152.5511169, 81.2579956, -233.2605896, 233.5207520
4: -140.0566101, 107.8988037, -140.5548248, 108.2794342, -248.3360443, 248.4536285
5: -125.1607285, 97.5191956, -125.6117477, 97.8683395, -223.0290527, 223.1309357
6: -120.2491989, 116.2837524, -120.6793823, 116.6907272, -236.9399261, 236.9631348
7: -130.6102905, 110.0433197, -131.0779724, 110.4361649, -241.0464325, 241.1212769
8: -158.8885803, 109.9428177, -159.4465179, 110.3260040, -269.2145996, 269.3893127
9: -119.2453156, 118.3672180, -119.6686935, 118.7806778, -238.0260010, 238.0359192

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5335490, upper bound: 202.5335490
time: 6.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5335490, upper bound: 202.5335490
time: 6.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -122.2579880, 97.7436218, -126.5763855, 101.1457443, -223.4037323, 224.3199768
1: -104.3062820, 86.7524567, -107.6923065, 89.7099686, -194.0162201, 194.4447632
2: -135.4862213, 87.9194107, -140.1774292, 91.0694809, -226.5556946, 228.0968323
3: -142.7414703, 76.1348343, -147.8384705, 78.7965698, -221.5380402, 223.9732819
4: -131.8283386, 101.5185394, -136.2753754, 105.0059280, -236.8342590, 237.7938843
5: -117.6028442, 91.6268845, -121.7529221, 94.8798294, -212.4826660, 213.3798065
6: -113.0164032, 109.6503143, -116.9965363, 113.1953888, -226.2117920, 226.6468506
7: -122.7404556, 103.4429855, -127.0589294, 107.0711746, -229.8116302, 230.5019226
8: -149.7062531, 103.4984512, -154.6617126, 107.0440063, -256.7502441, 258.1601562
9: -112.2027359, 111.5379715, -116.0323792, 115.2291718, -227.4319000, 227.5703430

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5288338, upper bound: 202.5288586
time: 7.08 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5293925, upper bound: 202.5293925
time: 6.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 1, lower bound: -202.5335490, upper bound: 202.5335490
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 1, lower bound: -202.5335490, upper bound: 202.5335490
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 1, lower bound: -202.5288338, upper bound: 202.5288586
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.68
Output dim: 1, lower bound: -202.5293925, upper bound: 202.5293925

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -130.1204224, 103.9453583, -130.1204224, 103.9453583, -234.0657806, 234.0657806
1: -110.6282425, 92.1818924, -110.6282425, 92.1818924, -202.8101349, 202.8101349
2: -144.0713348, 93.5925598, -144.0713348, 93.5925598, -237.6638947, 237.6638947
3: -152.0025940, 80.9696350, -152.0025940, 80.9696350, -232.9722290, 232.9722290
4: -140.0566101, 107.8988037, -140.0566101, 107.8988037, -247.9554138, 247.9554138
5: -125.1607285, 97.5191956, -125.1607285, 97.5191956, -222.6799011, 222.6799011
6: -120.2491989, 116.2837524, -120.2491989, 116.2837524, -236.5329590, 236.5329590
7: -130.6102905, 110.0433197, -130.6102905, 110.0433197, -240.6535950, 240.6535950
8: -158.8885803, 109.9428177, -158.8885803, 109.9428177, -268.8313599, 268.8313599
9: -119.2453156, 118.3672180, -119.2453156, 118.3672180, -237.6125336, 237.6125336

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5236365, upper bound: 202.5223506
time: 9.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5235396, upper bound: 202.5222624
time: 18.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -130.1204224, 103.9453583, -122.2579880, 97.7436218, -227.8639984, 226.2033386
1: -110.6282425, 92.1818924, -104.3062820, 86.7524567, -197.3807068, 196.4881744
2: -144.0713348, 93.5925598, -135.4862213, 87.9194107, -231.9907379, 229.0787811
3: -152.0025940, 80.9696350, -142.7414703, 76.1348343, -228.1374207, 223.7111053
4: -140.0566101, 107.8988037, -131.8283386, 101.5185394, -241.5751190, 239.7271423
5: -125.1607285, 97.5191956, -117.6028442, 91.6268845, -216.7876129, 215.1220398
6: -120.2491989, 116.2837524, -113.0164032, 109.6503143, -229.8995056, 229.3001556
7: -130.6102905, 110.0433197, -122.7404556, 103.4429855, -234.0532684, 232.7837524
8: -158.8885803, 109.9428177, -149.7062531, 103.4984512, -262.3870239, 259.6490479
9: -119.2453156, 118.3672180, -112.2027359, 111.5379715, -230.7832947, 230.5699463

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5331871, upper bound: 202.5320027
time: 10.96 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5338899, upper bound: 202.5327569
time: 8.39 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -113.6675262, 90.9790039, -114.4716110, 91.6141815, -205.2817078, 205.4506073
1: -97.1146774, 80.6771088, -97.5619888, 81.1579361, -178.2726135, 178.2391052
2: -125.9324112, 81.7306671, -126.7112656, 82.3533325, -208.2857361, 208.4419250
3: -132.5863037, 70.7420502, -133.5429840, 71.2023468, -203.7886505, 204.2850342
4: -122.7230606, 94.4404373, -123.4553757, 95.0350037, -217.7580566, 217.8958130
5: -109.3499298, 85.2708435, -110.1256409, 85.9249191, -195.2748260, 195.3964844
6: -105.1083908, 102.0896835, -105.8620071, 102.5462418, -207.6546326, 207.9516907
7: -114.0540619, 96.1754990, -114.8204041, 96.8350220, -210.8890839, 210.9959106
8: -139.3609009, 96.4174500, -140.0981445, 97.0737305, -236.4346161, 236.5155792
9: -104.3896790, 103.8449783, -105.0262680, 104.3937836, -208.7834625, 208.8712463

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5287960, upper bound: 202.5287960
time: 5.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5287960, upper bound: 202.5288586
time: 6.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -117.7201080, 94.1659393, -120.3384323, 96.2254181, -213.9455109, 214.5043640
1: -100.5257339, 83.5395889, -102.4946747, 85.2952194, -185.8209534, 186.0342712
2: -130.4433289, 84.6487961, -133.2530975, 86.5841522, -217.0274811, 217.9018860
3: -137.3928223, 73.2768555, -140.4859772, 74.8692627, -212.2620850, 213.7628326
4: -127.0216980, 97.7914200, -129.6640930, 99.8842621, -226.9059296, 227.4555054
5: -113.2433701, 88.2716064, -115.7621307, 90.2636948, -203.5070343, 204.0337372
6: -108.8370056, 105.6691284, -111.2615356, 107.7147522, -216.5517426, 216.9306641
7: -118.1567307, 99.6096497, -120.7645416, 101.8085556, -219.9652863, 220.3741760
8: -144.2477570, 99.7556534, -147.1574860, 101.9073944, -246.1551208, 246.9131470
9: -108.0810547, 107.4861526, -110.3684616, 109.6574097, -217.7384186, 217.8545990

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5288586, upper bound: 202.5288338
time: 6.15 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5288586, upper bound: 202.5293925
time: 6.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.03 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5236365, upper bound: 202.5223506
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5235396, upper bound: 202.5222624
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5331871, upper bound: 202.5320027
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5338899, upper bound: 202.5327569
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5287960, upper bound: 202.5287960
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5287960, upper bound: 202.5288586
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5288586, upper bound: 202.5288338
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 1, lower bound: -202.5288586, upper bound: 202.5293925

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -123.3892899, 98.6045456, -125.3437271, 100.1555939, -223.5448761, 223.9482727
1: -105.0109177, 87.4335022, -106.6423416, 88.8125839, -193.8235016, 194.0758362
2: -136.6441650, 88.7584381, -138.8007507, 90.1621246, -226.8062897, 227.5591888
3: -143.9888611, 76.7937469, -146.3154602, 78.0070038, -221.9958649, 223.1091766
4: -132.8001251, 102.3928146, -134.9073486, 103.9919662, -236.7920837, 237.3001556
5: -118.6462936, 92.4533081, -120.5379944, 93.9246750, -212.5709534, 212.9912872
6: -114.0494995, 110.3255463, -115.8503265, 112.0557480, -226.1052551, 226.1758575
7: -123.8310318, 104.3648453, -125.7997284, 106.0135880, -229.8446198, 230.1645813
8: -150.8370667, 104.4707642, -153.1753998, 106.0604095, -256.8973999, 257.6461487
9: -113.1262054, 112.3096237, -114.9034653, 114.0687103, -227.1949158, 227.2130585

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 121

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013314
time: 9.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013314
time: 9.43 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -126.1720047, 100.8480072, -123.5238190, 98.7249680, -224.8969269, 224.3718109
1: -107.3511734, 89.3771439, -105.1147537, 87.5287399, -194.8799133, 194.4918671
2: -139.7015991, 90.6966171, -136.7897797, 88.8423233, -228.5439148, 227.4863892
3: -147.2398987, 78.4893112, -144.1547241, 76.8875961, -224.1275024, 222.6440430
4: -135.7645264, 104.6321335, -132.9305573, 102.4840240, -238.2485352, 237.5626678
5: -121.3236847, 94.4857941, -118.7913361, 92.5496902, -213.8733826, 213.2771149
6: -116.6095963, 112.7886887, -114.1697159, 110.4353104, -227.0449066, 226.9583740
7: -126.5843506, 106.6970215, -123.9499207, 104.4743729, -231.0587158, 230.6469116
8: -154.2463837, 106.7671509, -150.9911652, 104.5831223, -258.8294983, 257.7583008
9: -115.6304855, 114.8045502, -113.2228928, 112.4198151, -228.0502930, 228.0274353

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013357
time: 9.79 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013357
time: 8.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -117.8881989, 94.3142776, -113.6675262, 90.9790039, -208.8671722, 207.9818115
1: -100.3918381, 83.5407639, -97.1146774, 80.6771088, -181.0689392, 180.6554413
2: -130.4613647, 84.7870026, -125.9324112, 81.7306671, -212.1920319, 210.7194061
3: -137.5596924, 73.2970200, -132.5863037, 70.7420502, -208.3017426, 205.8833313
4: -127.1032867, 97.8240051, -122.7230606, 94.4404373, -221.5437164, 220.5470428
5: -113.4119873, 88.4718170, -109.3499298, 85.2708435, -198.6828308, 197.8217316
6: -108.9990311, 105.5229797, -105.1083908, 102.0896835, -211.0887146, 210.6313782
7: -118.2432480, 99.7016983, -114.0540619, 96.1754990, -214.4187469, 213.7557526
8: -144.1730499, 99.8656464, -139.3609009, 96.4174500, -240.5904846, 239.2265472
9: -108.1235886, 107.4182053, -104.3896790, 103.8449783, -211.9685669, 211.8078918

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5331012, upper bound: 202.5319283
time: 9.15 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5331012, upper bound: 202.5320027
time: 10.01 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -123.9516373, 99.0798645, -117.7201080, 94.1659393, -218.1175537, 216.7999725
1: -105.4870987, 87.8160629, -100.5257339, 83.5395889, -189.0266724, 188.3417969
2: -137.2237396, 89.1582108, -130.4433289, 84.6487961, -221.8725281, 219.6015320
3: -144.7315979, 77.0875549, -137.3928223, 73.2768555, -218.0084534, 214.4803772
4: -133.5183411, 102.8319778, -127.0216980, 97.7914200, -231.3097534, 229.8536377
5: -119.2369308, 92.9547424, -113.2433701, 88.2716064, -207.5085449, 206.1981049
6: -114.5787659, 110.8622437, -108.8370056, 105.6691284, -220.2478790, 219.6992340
7: -124.3865509, 104.8400803, -118.1567307, 99.6096497, -223.9962006, 222.9968109
8: -151.4652557, 104.8628769, -144.2477570, 99.7556534, -251.2209015, 249.1106262
9: -113.6436157, 112.8560486, -108.0810547, 107.4861526, -221.1297607, 220.9371033

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 54

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5333079, upper bound: 202.5321828
time: 10.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5333079, upper bound: 202.5321828
time: 10.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -110.8644638, 88.7690735, -114.4716110, 91.6141815, -202.4786224, 203.2406769
1: -94.7788391, 78.6979446, -97.5619888, 81.1579361, -175.9367676, 176.2599335
2: -122.8148117, 79.7017746, -126.7112656, 82.3533325, -205.1681366, 206.4130402
3: -129.2755280, 68.9802322, -133.5429840, 71.2023468, -200.4778748, 202.5232239
4: -119.7650375, 92.1343384, -123.4553757, 95.0350037, -214.8000488, 215.5897217
5: -106.6543579, 83.1968613, -110.1256409, 85.9249191, -192.5792694, 193.3225098
6: -102.5259552, 99.6366196, -105.8620071, 102.5462418, -205.0722046, 205.4986267
7: -111.2198029, 93.8018799, -114.8204041, 96.8350220, -208.0548248, 208.6222534
8: -135.9983368, 94.1124878, -140.0981445, 97.0737305, -233.0720673, 234.2106171
9: -101.8483963, 101.3442841, -105.0262680, 104.3937836, -206.2421722, 206.3705444

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5239308, upper bound: 202.5240268
time: 6.64 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
time: 6.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -115.8882217, 92.7210236, -114.4716110, 91.6141815, -207.5023956, 207.1926270
1: -99.0006104, 82.2422943, -97.5619888, 81.1579361, -180.1585388, 179.8042908
2: -128.4081726, 83.3292465, -126.7112656, 82.3533325, -210.7614899, 210.0404968
3: -135.2324066, 72.1234207, -133.5429840, 71.2023468, -206.4347534, 205.6663971
4: -125.0786591, 96.2876358, -123.4553757, 95.0350037, -220.1136627, 219.7430115
5: -111.4838257, 86.9164505, -110.1256409, 85.9249191, -197.4087067, 197.0420837
6: -107.1486588, 104.0609818, -105.8620071, 102.5462418, -209.6949005, 209.9229889
7: -116.3060379, 98.0624924, -114.8204041, 96.8350220, -213.1410522, 212.8828888
8: -142.0432129, 98.2453613, -140.0981445, 97.0737305, -239.1169434, 238.3435059
9: -106.4165268, 105.8507843, -105.0262680, 104.3937836, -210.8103027, 210.8770447

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5082526, upper bound: 202.5084614
time: 7.32 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5074190, upper bound: 202.5074193
time: 6.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -110.8644638, 88.7690735, -120.3384323, 96.2254181, -207.0898590, 209.1074829
1: -94.7788391, 78.6979446, -102.4946747, 85.2952194, -180.0740509, 181.1926117
2: -122.8148117, 79.7017746, -133.2530975, 86.5841522, -209.3989563, 212.9548645
3: -129.2755280, 68.9802322, -140.4859772, 74.8692627, -204.1447601, 209.4662018
4: -119.7650375, 92.1343384, -129.6640930, 99.8842621, -219.6492920, 221.7984314
5: -106.6543579, 83.1968613, -115.7621307, 90.2636948, -196.9180450, 198.9589844
6: -102.5259552, 99.6366196, -111.2615356, 107.7147522, -210.2406921, 210.8981628
7: -111.2198029, 93.8018799, -120.7645416, 101.8085556, -213.0283508, 214.5663910
8: -135.9983368, 94.1124878, -147.1574860, 101.9073944, -237.9057007, 241.2699738
9: -101.8483963, 101.3442841, -110.3684616, 109.6574097, -211.5057526, 211.7127380

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5239308, upper bound: 202.5240601
time: 6.88 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5238368
time: 6.11 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -115.9179535, 92.7447739, -120.3384323, 96.2254181, -212.1433563, 213.0831909
1: -99.0251465, 82.2631683, -102.4946747, 85.2952194, -184.3203583, 184.7578430
2: -128.4415894, 83.3498840, -133.2530975, 86.5841522, -215.0257416, 216.6029816
3: -135.2681427, 72.1417007, -140.4859772, 74.8692627, -210.1373901, 212.6276855
4: -125.1119843, 96.3117828, -129.6640930, 99.8842621, -224.9962463, 225.9758759
5: -111.5123749, 86.9388809, -115.7621307, 90.2636948, -201.7760620, 202.7010193
6: -107.1769104, 104.0880127, -111.2615356, 107.7147522, -214.8916321, 215.3495483
7: -116.3367844, 98.0876541, -120.7645416, 101.8085556, -218.1453400, 218.8521881
8: -142.0798798, 98.2697754, -147.1574860, 101.9073944, -243.9872437, 245.4272614
9: -106.4445038, 105.8768539, -110.3684616, 109.6574097, -216.1018677, 216.2452850

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5125360, upper bound: 202.5141688
time: 6.92 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5120630, upper bound: 202.5136263
time: 6.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.54 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013314
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013314
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013357
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.6013314, upper bound: 202.6013357
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5331012, upper bound: 202.5319283
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5331012, upper bound: 202.5320027
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5333079, upper bound: 202.5321828
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5333079, upper bound: 202.5321828
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5239308, upper bound: 202.5240268
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5082526, upper bound: 202.5084614
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5074190, upper bound: 202.5074193
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5239308, upper bound: 202.5240601
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5238368
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5125360, upper bound: 202.5141688
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.54
Output dim: 1, lower bound: -202.5120630, upper bound: 202.5136263

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -123.3892899, 98.6045456, -123.3892899, 98.6045456, -221.9938354, 221.9938354
1: -105.0109177, 87.4335022, -105.0109177, 87.4335022, -192.4444122, 192.4444275
2: -136.6441650, 88.7584381, -136.6441650, 88.7584381, -225.4026031, 225.4026031
3: -143.9888611, 76.7937469, -143.9888611, 76.7937469, -220.7825928, 220.7825928
4: -132.8001251, 102.3928146, -132.8001251, 102.3928146, -235.1929321, 235.1929321
5: -118.6462936, 92.4533081, -118.6462936, 92.4533081, -211.0995941, 211.0995941
6: -114.0494995, 110.3255463, -114.0494995, 110.3255463, -224.3750458, 224.3750458
7: -123.8310318, 104.3648453, -123.8310318, 104.3648453, -228.1958771, 228.1958771
8: -150.8370667, 104.4707642, -150.8370667, 104.4707642, -255.3078308, 255.3078308
9: -113.1262054, 112.3096237, -113.1262054, 112.3096237, -225.4358215, 225.4358215

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5013612, upper bound: 202.5001514
time: 10.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4980059, upper bound: 202.4974324
time: 9.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -123.3892899, 98.6045456, -126.1720047, 100.8480072, -224.2372894, 224.7765503
1: -105.0109177, 87.4335022, -107.3511734, 89.3771439, -194.3880615, 194.7846680
2: -136.6441650, 88.7584381, -139.7015991, 90.6966171, -227.3407898, 228.4600372
3: -143.9888611, 76.7937469, -147.2398987, 78.4893112, -222.4781799, 224.0336151
4: -132.8001251, 102.3928146, -135.7645264, 104.6321335, -237.4322510, 238.1573334
5: -118.6462936, 92.4533081, -121.3236847, 94.4857941, -213.1320801, 213.7769928
6: -114.0494995, 110.3255463, -116.6095963, 112.7886887, -226.8381805, 226.9351501
7: -123.8310318, 104.3648453, -126.5843506, 106.6970215, -230.5280457, 230.9491882
8: -150.8370667, 104.4707642, -154.2463837, 106.7671509, -257.6041565, 258.7171631
9: -113.1262054, 112.3096237, -115.6304855, 114.8045502, -227.9307556, 227.9400940

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5216651, upper bound: 202.5220826
time: 9.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4980059, upper bound: 202.4974324
time: 8.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -126.1720047, 100.8480072, -123.3892899, 98.6045456, -224.7765503, 224.2372894
1: -107.3511734, 89.3771439, -105.0109177, 87.4335022, -194.7846680, 194.3880615
2: -139.7015991, 90.6966171, -136.6441650, 88.7584381, -228.4600372, 227.3407898
3: -147.2398987, 78.4893112, -143.9888611, 76.7937469, -224.0336151, 222.4781799
4: -135.7645264, 104.6321335, -132.8001251, 102.3928146, -238.1573334, 237.4322510
5: -121.3236847, 94.4857941, -118.6462936, 92.4533081, -213.7769928, 213.1320801
6: -116.6095963, 112.7886887, -114.0494995, 110.3255463, -226.9351501, 226.8381805
7: -126.5843506, 106.6970215, -123.8310318, 104.3648453, -230.9491882, 230.5280457
8: -154.2463837, 106.7671509, -150.8370667, 104.4707642, -258.7171631, 257.6041565
9: -115.6304855, 114.8045502, -113.1262054, 112.3096237, -227.9400940, 227.9307556

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5012675, upper bound: 202.5000611
time: 9.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4950703, upper bound: 202.4950703
time: 6.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -126.1720047, 100.8480072, -126.1720047, 100.8480072, -227.0200043, 227.0200043
1: -107.3511734, 89.3771439, -107.3511734, 89.3771439, -196.7283173, 196.7283173
2: -139.7015991, 90.6966171, -139.7015991, 90.6966171, -230.3982239, 230.3982239
3: -147.2398987, 78.4893112, -147.2398987, 78.4893112, -225.7292175, 225.7292175
4: -135.7645264, 104.6321335, -135.7645264, 104.6321335, -240.3966522, 240.3966522
5: -121.3236847, 94.4857941, -121.3236847, 94.4857941, -215.8094788, 215.8094788
6: -116.6095963, 112.7886887, -116.6095963, 112.7886887, -229.3982849, 229.3982849
7: -126.5843506, 106.6970215, -126.5843506, 106.6970215, -233.2813721, 233.2813721
8: -154.2463837, 106.7671509, -154.2463837, 106.7671509, -261.0135498, 261.0135498
9: -115.6304855, 114.8045502, -115.6304855, 114.8045502, -230.4350281, 230.4350281

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5012675, upper bound: 202.5000611
time: 8.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.4950703, upper bound: 202.4950703
time: 6.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -117.8881989, 94.3142776, -110.8644638, 88.7690735, -206.6572266, 205.1787415
1: -100.3918381, 83.5407639, -94.7788391, 78.6979446, -179.0897827, 178.3196106
2: -130.4613647, 84.7870026, -122.8148117, 79.7017746, -210.1631470, 207.6017761
3: -137.5596924, 73.2970200, -129.2755280, 68.9802322, -206.5399170, 202.5725403
4: -127.1032867, 97.8240051, -119.7650375, 92.1343384, -219.2376251, 217.5890198
5: -113.4119873, 88.4718170, -106.6543579, 83.1968613, -196.6088409, 195.1261749
6: -108.9990311, 105.5229797, -102.5259552, 99.6366196, -208.6356354, 208.0489349
7: -118.2432480, 99.7016983, -111.2198029, 93.8018799, -212.0451355, 210.9214935
8: -144.1730499, 99.8656464, -135.9983368, 94.1124878, -238.2855225, 235.8639832
9: -108.1235886, 107.4182053, -101.8483963, 101.3442841, -209.4678650, 209.2666016

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5281689, upper bound: 202.5270093
time: 8.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5281352, upper bound: 202.5269806
time: 9.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -117.8881989, 94.3142776, -115.8882217, 92.7210236, -210.6091919, 210.2024994
1: -100.3918381, 83.5407639, -99.0006104, 82.2422943, -182.6341248, 182.5413818
2: -130.4613647, 84.7870026, -128.4081726, 83.3292465, -213.7906189, 213.1951294
3: -137.5596924, 73.2970200, -135.2324066, 72.1234207, -209.6831055, 208.5294189
4: -127.1032867, 97.8240051, -125.0786591, 96.2876358, -223.3909149, 222.9026489
5: -113.4119873, 88.4718170, -111.4838257, 86.9164505, -200.3284302, 199.9556274
6: -108.9990311, 105.5229797, -107.1486588, 104.0609818, -213.0600128, 212.6716309
7: -118.2432480, 99.7016983, -116.3060379, 98.0624924, -216.3057404, 216.0077362
8: -144.1730499, 99.8656464, -142.0432129, 98.2453613, -242.4184113, 241.9088593
9: -108.1235886, 107.4182053, -106.4165268, 105.8507843, -213.9743652, 213.8347321

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 54

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5123815, upper bound: 202.5113131
time: 9.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5120184, upper bound: 202.5109480
time: 9.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -123.9516373, 99.0798645, -110.8644638, 88.7690735, -212.7206573, 209.9443359
1: -105.4870987, 87.8160629, -94.7788391, 78.6979446, -184.1850128, 182.5949097
2: -137.2237396, 89.1582108, -122.8148117, 79.7017746, -216.9255066, 211.9730225
3: -144.7315979, 77.0875549, -129.2755280, 68.9802322, -213.7118225, 206.3630524
4: -133.5183411, 102.8319778, -119.7650375, 92.1343384, -225.6526794, 222.5970154
5: -119.2369308, 92.9547424, -106.6543579, 83.1968613, -202.4337921, 199.6091003
6: -114.5787659, 110.8622437, -102.5259552, 99.6366196, -214.2153778, 213.3881989
7: -124.3865509, 104.8400803, -111.2198029, 93.8018799, -218.1884155, 216.0598755
8: -151.4652557, 104.8628769, -135.9983368, 94.1124878, -245.5777283, 240.8612061
9: -113.6436157, 112.8560486, -101.8483963, 101.3442841, -214.9878998, 214.7044373

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5288569, upper bound: 202.5277651
time: 10.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5288029, upper bound: 202.5277257
time: 10.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -123.9516373, 99.0798645, -115.9179535, 92.7447739, -216.6963654, 214.9978180
1: -105.4870987, 87.8160629, -99.0251465, 82.2631683, -187.7502594, 186.8412170
2: -137.2237396, 89.1582108, -128.4415894, 83.3498840, -220.5736237, 217.5997925
3: -144.7315979, 77.0875549, -135.2681427, 72.1417007, -216.8732910, 212.3556824
4: -133.5183411, 102.8319778, -125.1119843, 96.3117828, -229.8301239, 227.9439697
5: -119.2369308, 92.9547424, -111.5123749, 86.9388809, -206.1758118, 204.4671173
6: -114.5787659, 110.8622437, -107.1769104, 104.0880127, -218.6667786, 218.0391388
7: -124.3865509, 104.8400803, -116.3367844, 98.0876541, -222.4742126, 221.1768646
8: -151.4652557, 104.8628769, -142.0798798, 98.2697754, -249.7350311, 246.9427490
9: -113.6436157, 112.8560486, -106.4445038, 105.8768539, -219.5204620, 219.3005524

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5172443, upper bound: 202.5174777
time: 9.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5171470, upper bound: 202.5173951
time: 10.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.9678879, 86.4815521, -110.3446426, 88.3557816, -196.3236694, 196.8261566
1: -92.3447723, 76.6546097, -94.0898514, 78.2469254, -170.5916901, 170.7444611
2: -119.6357498, 77.6426926, -122.1837387, 79.4210663, -199.0568085, 199.8264313
3: -125.8546982, 67.1702042, -128.6736755, 68.6212616, -194.4759369, 195.8438721
4: -116.6752701, 89.7615891, -119.0504608, 91.6561966, -208.3314667, 208.8120422
5: -103.8556290, 81.0367432, -106.1387863, 82.8473129, -186.7029266, 187.1755371
6: -99.8573761, 97.0885162, -102.0611420, 98.9129105, -198.7702942, 199.1496582
7: -108.3341827, 91.3711929, -110.7090073, 93.3742447, -201.7084198, 202.0801849
8: -132.5180969, 91.7641220, -135.1403351, 93.7227554, -226.2408447, 226.9044495
9: -99.2238617, 98.7449570, -101.2834244, 100.6914825, -199.9153290, 200.0283508

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 54

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
time: 7.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
time: 7.27 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -108.1678238, 86.6399918, -113.4731979, 90.8484039, -199.0162354, 200.1131897
1: -92.5110397, 76.7934875, -96.7147751, 80.4502029, -172.9612274, 173.5082245
2: -119.8538666, 77.7781219, -125.6299210, 81.6365814, -201.4904327, 203.4080505
3: -126.0880661, 67.2991791, -132.3530121, 70.5475311, -196.6355743, 199.6521759
4: -116.8916931, 89.9230881, -122.4217987, 94.2222519, -211.1139526, 212.3448792
5: -104.0444336, 81.1927109, -109.1529770, 85.1885071, -189.2329407, 190.3456879
6: -100.0373688, 97.2669373, -104.9315338, 101.6819458, -201.7193146, 202.1984711
7: -108.5310974, 91.5363770, -113.8299026, 95.9991302, -204.5302277, 205.3662415
8: -132.7529907, 91.9308167, -138.9157867, 96.2833405, -229.0363312, 230.8466034
9: -99.4102554, 98.9249115, -104.1313858, 103.4859009, -202.8961487, 203.0563049

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 54

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
time: 6.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
time: 7.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.28 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5013612, upper bound: 202.5001514
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.4980059, upper bound: 202.4974324
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5216651, upper bound: 202.5220826
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.4980059, upper bound: 202.4974324
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5012675, upper bound: 202.5000611
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.4950703, upper bound: 202.4950703
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5012675, upper bound: 202.5000611
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.4950703, upper bound: 202.4950703
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5281689, upper bound: 202.5270093
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5281352, upper bound: 202.5269806
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5123815, upper bound: 202.5113131
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5120184, upper bound: 202.5109480
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5288569, upper bound: 202.5277651
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5288029, upper bound: 202.5277257
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5172443, upper bound: 202.5174777
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5171470, upper bound: 202.5173951
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.28
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5237783
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 1, lower bound: -202.5082526, upper bound: 202.5084614
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 1, lower bound: -202.5074190, upper bound: 202.5074193
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 1, lower bound: -202.5239308, upper bound: 202.5240601
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 1, lower bound: -202.5237783, upper bound: 202.5238368
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 1, lower bound: -202.5125360, upper bound: 202.5141688
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.28
Output dim: 1, lower bound: -202.5120630, upper bound: 202.5136263
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=203.5233612060547
rel_dist={1: [-202.608202498089, 202.60820249808899]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1814.21 seconds
