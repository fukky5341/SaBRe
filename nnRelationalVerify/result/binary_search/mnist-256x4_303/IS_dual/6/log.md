## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 372.218729681
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213)
1: (-173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792)
2: (-227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798)
3: (-241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816)
4: (-221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640)
5: (-198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285)
6: (-190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555)
7: (-206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513)
8: (-249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819)
9: (-188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024)

## BASE Result
execution time: IAR + LP analysis = 1.02 + 10.85 = 11.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -372.2698839, upper bound: 372.2698839


# Binary Search by BASE starts (time budget: 2688.13 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=375.07427978515625
rel_dist={2: [-372.26984149889176, 372.2698414979088]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=375.07427978515625
rel_dist={2: [-372.2698094205705, 372.2698094230649]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=375.07427978515625
rel_dist={2: [-372.26978283690505, 372.26978280960407]}

## Binary Search Result
Binary search time: 46.61 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2641.52 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2597993, upper bound: 372.2579874
time: 7.78 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2560994, upper bound: 372.2560993
time: 7.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.21 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 15.21
Output dim: 2, lower bound: -372.2597993, upper bound: 372.2579874
IS_B2, status: Status.UNKNOWN, split count: 1, time: 15.21
Output dim: 2, lower bound: -372.2560994, upper bound: 372.2560993

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -206.2513123, 163.9280243, -198.4893188, 157.8231659, -364.0744629, 362.4172668
1: -173.1095734, 145.2978516, -166.5716248, 139.8642120, -312.9736938, 311.8694763
2: -227.4908142, 147.5834808, -218.9606934, 142.0608978, -369.5516968, 366.5441895
3: -241.3278656, 127.4458313, -232.2102203, 122.6454620, -363.9733276, 359.6559753
4: -221.8932953, 169.6110992, -213.5940399, 163.2548523, -385.1481323, 383.2051392
5: -198.0002136, 153.7202454, -190.5393372, 147.9361877, -345.9364014, 344.2595825
6: -190.1141815, 182.9126740, -182.9729309, 176.0473022, -366.1614990, 365.8856201
7: -206.7764435, 173.4635925, -198.9613647, 166.9206238, -373.6970825, 372.4249268
8: -249.4098511, 171.1091919, -240.0978241, 164.7572479, -414.1670837, 411.2069397
9: -188.2842560, 185.4169464, -181.2092133, 178.4702759, -366.7545166, 366.6261597

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2522078, upper bound: 372.2486203
time: 9.14 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2564346, upper bound: 372.2543306
time: 8.44 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -203.6210022, 161.8539276, -199.2478333, 158.4422455, -362.0632324, 361.1017151
1: -170.8947144, 143.4514008, -167.1326599, 140.3604584, -311.2551575, 310.5839844
2: -224.5964050, 145.7047119, -219.7879791, 142.5113831, -367.1077881, 365.4926758
3: -238.2326660, 125.8168182, -233.0491180, 123.0219116, -361.2545471, 358.8659363
4: -219.0767212, 167.4510803, -214.4481812, 163.7952576, -382.8719788, 381.8992615
5: -195.4674683, 151.7577667, -191.1925201, 148.4223480, -343.8898010, 342.9502869
6: -187.6946869, 180.5843201, -183.6501465, 176.7235718, -364.4182129, 364.2344666
7: -204.1248779, 171.2452850, -199.6473389, 167.4870148, -371.6118774, 370.8926392
8: -246.2509766, 168.9484253, -241.0108185, 165.3170319, -411.5679932, 409.9592285
9: -185.8880157, 183.0614166, -181.8820496, 179.1553345, -365.0433350, 364.9434204

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 104

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2482098, upper bound: 372.2466146
time: 9.87 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523194, upper bound: 372.2523194
time: 8.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.40 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 19.40
Output dim: 2, lower bound: -372.2522078, upper bound: 372.2486203
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 19.40
Output dim: 2, lower bound: -372.2564346, upper bound: 372.2543306
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 19.40
Output dim: 2, lower bound: -372.2482098, upper bound: 372.2466146
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 19.40
Output dim: 2, lower bound: -372.2523194, upper bound: 372.2523194

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -203.8014679, 161.9264679, -197.4121704, 156.9631653, -360.7645874, 359.3386230
1: -171.1221161, 143.5572357, -165.6781616, 139.1048126, -310.2268677, 309.2354126
2: -224.8100433, 145.7190399, -217.7776947, 141.2812958, -366.0912781, 363.4967346
3: -238.4075775, 125.9816818, -230.9389954, 121.9907455, -360.3983154, 356.9206848
4: -219.2710876, 167.5274048, -212.4391174, 162.3628082, -381.6338501, 379.9665222
5: -195.5661774, 151.8196869, -189.4939270, 147.1280060, -342.6940918, 341.3135986
6: -187.8642273, 180.8197479, -181.9837036, 175.1055450, -362.9697266, 362.8034668
7: -204.2772522, 171.3745575, -197.8777771, 166.0152740, -370.2925415, 369.2523193
8: -246.5964355, 169.0711212, -238.8204041, 163.8702087, -410.4665833, 407.8914795
9: -186.0519562, 183.2635956, -180.2310638, 177.5119934, -363.5639038, 363.4946594

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 204

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1761797, upper bound: 372.1890469
time: 10.57 seconds

## Relational analysis of IS_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1832505, upper bound: 372.1962112
time: 8.47 seconds

## Relational analysis of IS_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2410937, upper bound: 372.2404510
time: 9.29 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2376399, upper bound: 372.2340417
time: 8.76 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -204.1797943, 162.2701569, -198.4893188, 157.8231659, -362.0029602, 360.7594299
1: -171.3950958, 143.8330078, -166.5716248, 139.8642120, -311.2592773, 310.4046326
2: -225.2109680, 146.0715637, -218.9606934, 142.0608978, -367.2718506, 365.0322266
3: -238.8656158, 126.1874695, -232.2102203, 122.6454620, -361.5110168, 358.3976440
4: -219.6696930, 167.8824158, -213.5940399, 163.2548523, -382.9245605, 381.4764404
5: -195.9801941, 152.1619873, -190.5393372, 147.9361877, -343.9163818, 342.7013245
6: -188.2082062, 181.1004944, -182.9729309, 176.0473022, -364.2554626, 364.0733643
7: -204.6786652, 171.7150421, -198.9613647, 166.9206238, -371.5992126, 370.6763916
8: -246.9599609, 169.3978729, -240.0978241, 164.7572479, -411.7172241, 409.4956665
9: -186.3996887, 183.5701141, -181.2092133, 178.4702759, -364.8699646, 364.7793274

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 204

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1868406, upper bound: 372.2010387
time: 8.42 seconds

## Relational analysis of IS_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1924596, upper bound: 372.2057499
time: 8.36 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1503546, upper bound: 372.1319622
time: 8.30 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -201.1930847, 159.8713684, -198.2055664, 157.6105652, -358.8036194, 358.0769043
1: -168.9250183, 141.7270813, -166.2677307, 139.6258087, -308.5507507, 307.9948120
2: -221.9417725, 143.8571320, -218.6437988, 141.7577362, -363.6994629, 362.5009155
3: -235.3393555, 124.3657913, -231.8200836, 122.3880920, -357.7274475, 356.1858521
4: -216.4794006, 165.3861542, -213.3305817, 162.9325409, -379.4118652, 378.7166748
5: -193.0557404, 149.8743286, -190.1817322, 147.6407776, -340.6965332, 340.0559998
6: -185.4648895, 178.5113068, -182.6930237, 175.8118439, -361.2767334, 361.2043152
7: -201.6495056, 169.1757202, -198.5995636, 166.6114960, -368.2609863, 367.7752686
8: -243.4635315, 166.9286652, -239.7742157, 164.4581757, -407.9216919, 406.7028503
9: -183.6757050, 180.9284363, -180.9350739, 178.2280121, -361.9036865, 361.8635254

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 104

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1376539, upper bound: 372.1679318
time: 8.21 seconds

## Relational analysis of IS_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2367222, upper bound: 372.2379778
time: 9.85 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2335769, upper bound: 372.2318602
time: 8.97 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -201.5388794, 160.1873322, -199.2478333, 158.4422455, -359.9811096, 359.4351807
1: -169.1715851, 141.9791870, -167.1326599, 140.3604584, -309.5320435, 309.1117859
2: -222.3047791, 144.1848450, -219.7879791, 142.5113831, -364.8161011, 363.9728394
3: -235.7579651, 124.5520935, -233.0491180, 123.0219116, -358.7798767, 357.6011963
4: -216.8419189, 165.7134705, -214.4481812, 163.7952576, -380.6371765, 380.1616516
5: -193.4369202, 150.1913757, -191.1925201, 148.4223480, -341.8592529, 341.3839111
6: -185.7788239, 178.7630768, -183.6501465, 176.7235718, -362.5023499, 362.4131470
7: -202.0161438, 169.4875946, -199.6473389, 167.4870148, -369.5031433, 369.1349487
8: -243.7885590, 167.2284698, -241.0108185, 165.3170319, -409.1055908, 408.2392883
9: -183.9938202, 181.2050781, -181.8820496, 179.1553345, -363.1491089, 363.0871277

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1431770, upper bound: 372.1769643
time: 7.07 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1204969, upper bound: 372.1204969
time: 5.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.85 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.2410937, upper bound: 372.2404510
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.2376399, upper bound: 372.2340417
IS_B1_A2_A1, status: Status.VERIFIED, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.1924596, upper bound: 372.2057499
IS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.1503546, upper bound: 372.1319622
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.2367222, upper bound: 372.2379778
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.2335769, upper bound: 372.2318602
IS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.1431770, upper bound: 372.1769643
IS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 15.85
Output dim: 2, lower bound: -372.1204969, upper bound: 372.1204969

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -199.8621826, 158.7796631, -197.4121704, 156.9631653, -356.8252869, 356.1917725
1: -167.7702026, 140.7504883, -165.6781616, 139.1048126, -306.8750000, 306.4286499
2: -220.4415588, 142.8511963, -217.7776947, 141.2812958, -361.7228088, 360.6289062
3: -233.7405548, 123.5096359, -230.9389954, 121.9907455, -355.7312927, 354.4486389
4: -215.0403290, 164.2514496, -212.4391174, 162.3628082, -377.4031372, 376.6905518
5: -191.7583008, 148.8931580, -189.4939270, 147.1280060, -338.8862305, 338.3870850
6: -184.2117310, 177.3080902, -181.9837036, 175.1055450, -359.3172607, 359.2917786
7: -200.2877197, 168.0493469, -197.8777771, 166.0152740, -366.3029785, 365.9271240
8: -241.7997589, 165.7384949, -238.8204041, 163.8702087, -405.6699219, 404.5588684
9: -182.4562073, 179.6923676, -180.2310638, 177.5119934, -359.9681396, 359.9234314

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 204

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2263653, upper bound: 372.2230199
time: 8.69 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197819, upper bound: 372.2195237
time: 8.16 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -203.5211945, 161.6500397, -195.9754944, 155.8165741, -359.3377686, 357.6255188
1: -170.7705994, 143.2739563, -164.4574127, 138.0834351, -308.8540344, 307.7313538
2: -224.4299774, 145.3036194, -216.1865082, 140.2349396, -364.6648865, 361.4901123
3: -238.0206909, 125.6587982, -229.2409363, 121.0914078, -359.1120911, 354.8997192
4: -218.9474640, 167.1878510, -210.8920135, 161.1688690, -380.1163330, 378.0798645
5: -195.2541046, 151.5824585, -188.1061707, 146.0605621, -341.3146362, 339.6885376
6: -187.5460052, 180.5043793, -180.6519775, 173.8256836, -361.3717041, 361.1563721
7: -203.8778992, 171.0730133, -196.4233093, 164.8027191, -368.6806030, 367.4963379
8: -246.1284485, 168.5435791, -237.0729828, 162.6553955, -408.7838440, 405.6165466
9: -185.7773590, 182.9204712, -178.9205627, 176.2114105, -361.9887695, 361.8409424

Time for backsubstitution: 0.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2225345, upper bound: 372.2146866
time: 8.51 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2160095, upper bound: 372.2117834
time: 7.50 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -197.2444611, 156.7172089, -198.2055664, 157.6105652, -354.8549500, 354.9227600
1: -165.5651855, 138.9138641, -166.2677307, 139.6258087, -305.1909180, 305.1815796
2: -217.5627136, 140.9822388, -218.6437988, 141.7577362, -359.3204346, 359.6260376
3: -230.6611023, 121.8880920, -231.8200836, 122.3880920, -353.0491943, 353.7081909
4: -212.2383118, 162.1026154, -213.3305817, 162.9325409, -375.1708374, 375.4331055
5: -189.2388153, 146.9407959, -190.1817322, 147.6407776, -336.8795471, 337.1224670
6: -181.8038635, 174.9908447, -182.6930237, 175.8118439, -357.6156921, 357.6838684
7: -197.6504211, 165.8424683, -198.5995636, 166.6114960, -364.2619019, 364.4420166
8: -238.6553345, 163.5875549, -239.7742157, 164.4581757, -403.1134338, 403.3617249
9: -180.0716248, 177.3483124, -180.9350739, 178.2280121, -358.2996216, 358.2833557

Time for backsubstitution: 0.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2199988, upper bound: 372.2247016
time: 8.66 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2167745, upper bound: 372.2182292
time: 8.44 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -200.9438019, 159.6194153, -196.8106842, 156.4972076, -357.4410095, 356.4300842
1: -168.6003265, 141.4659882, -165.0825043, 138.6330414, -307.2333679, 306.5484924
2: -221.5964355, 143.4656525, -217.0986328, 140.7422180, -362.3386230, 360.5642395
3: -234.9898224, 124.0622482, -230.1709290, 121.5143967, -356.5042114, 354.2331848
4: -216.1897125, 165.0730286, -211.8283691, 161.7731476, -377.9627991, 376.9013977
5: -192.7734375, 149.6613159, -188.8338165, 146.6047058, -339.3781433, 338.4951172
6: -185.1760559, 178.2241974, -181.3997650, 174.5690460, -359.7451172, 359.6239624
7: -201.2833405, 168.9009094, -197.1881714, 165.4343872, -366.7175903, 366.0890808
8: -243.0346069, 166.4282227, -238.0774384, 163.2786865, -406.3132935, 404.5056763
9: -183.4300079, 180.6138306, -179.6619720, 176.9650574, -360.3950500, 360.2758179

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 104

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2170114, upper bound: 372.2181664
time: 8.57 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2132993, upper bound: 372.2105408
time: 8.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.04 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2263653, upper bound: 372.2230199
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2197819, upper bound: 372.2195237
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2225345, upper bound: 372.2146866
IS_B1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2160095, upper bound: 372.2117834
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2199988, upper bound: 372.2247016
IS_B2_A1_A1_A2, status: Status.VERIFIED, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2167745, upper bound: 372.2182292
IS_B2_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2170114, upper bound: 372.2181664
IS_B2_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 26.04
Output dim: 2, lower bound: -372.2132993, upper bound: 372.2105408

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -199.8621826, 158.7796631, -188.1694489, 149.6298676, -349.4920349, 346.9490662
1: -167.7702026, 140.7504883, -157.8686676, 132.5978699, -300.3680420, 298.6191406
2: -220.4415588, 142.8511963, -207.5942078, 134.6568604, -355.0983582, 350.4453735
3: -233.7405548, 123.5096359, -220.0727844, 116.2802658, -350.0208130, 343.5823669
4: -215.0403290, 164.2514496, -202.5443420, 154.7434998, -369.7838135, 366.7957764
5: -191.7583008, 148.8931580, -180.5824432, 140.2569885, -332.0152588, 329.4755859
6: -184.2117310, 177.3080902, -173.4409637, 166.9221344, -351.1338501, 350.7489624
7: -200.2877197, 168.0493469, -188.6085815, 158.2368011, -358.5245361, 356.6579285
8: -241.7997589, 165.7384949, -227.6616516, 156.1918030, -397.9915771, 393.4001465
9: -182.4562073, 179.6923676, -171.8278351, 169.2462158, -351.7023010, 351.5202026

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2263653, upper bound: 372.2230199
time: 9.33 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2263653, upper bound: 372.2230199
time: 9.07 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -196.1069336, 155.7970734, -180.8238831, 143.7690582, -339.8759766, 336.6208801
1: -164.5954132, 138.1034698, -151.5677948, 127.3773727, -291.9727478, 289.6712341
2: -216.3059692, 140.1540833, -199.4961395, 129.2515869, -345.5575562, 339.6502075
3: -229.3238068, 121.1875153, -211.4344482, 111.6564026, -340.9802246, 332.6219482
4: -211.0090942, 161.1455994, -194.6394196, 148.5626373, -359.5717163, 355.7850342
5: -188.1337433, 146.0997925, -173.4427185, 134.7452240, -322.8789673, 319.5424805
6: -180.7360229, 173.9833832, -166.5958862, 160.4036865, -341.1395874, 340.5792847
7: -196.5239410, 164.8915253, -181.1964569, 152.0325012, -348.5564575, 346.0879822
8: -237.2568665, 162.6000977, -218.7143555, 149.8806458, -387.1375122, 381.3144531
9: -179.0459137, 176.3326416, -165.1796265, 162.6873016, -341.7332153, 341.5122681

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 104

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197819, upper bound: 372.2195237
time: 9.36 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197819, upper bound: 372.2195237
time: 8.23 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -203.5211945, 161.6500397, -186.7177887, 148.4714203, -351.9926147, 348.3678284
1: -170.7705994, 143.2739563, -156.6354065, 131.5655060, -302.3361206, 299.9093628
2: -224.4299774, 145.3036194, -205.9865265, 133.5991669, -358.0290527, 351.2900696
3: -238.0206909, 125.6587982, -218.3568268, 115.3714905, -353.3921814, 344.0156250
4: -218.9474640, 167.1878510, -200.9808350, 153.5369568, -372.4844360, 368.1687012
5: -195.2541046, 151.5824585, -179.1801300, 139.1785126, -334.4326172, 330.7625427
6: -187.5460052, 180.5043793, -172.0952454, 165.6292877, -353.1752930, 352.5996094
7: -203.8778992, 171.0730133, -187.1388550, 157.0117188, -360.8895264, 358.2118530
8: -246.1284485, 168.5435791, -225.8962708, 154.9640961, -401.0925293, 394.4397888
9: -185.7773590, 182.9204712, -170.5035553, 167.9323730, -353.7097168, 353.4240112

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2225345, upper bound: 372.2146866
time: 9.21 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2225345, upper bound: 372.2146866
time: 9.28 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -187.7964630, 149.2179718, -198.2055664, 157.6105652, -345.4070129, 347.4234924
1: -157.5846863, 132.2639771, -166.2677307, 139.6258087, -297.2103882, 298.5316467
2: -207.1527405, 134.2106628, -218.6437988, 141.7577362, -348.9104309, 352.8544617
3: -219.5525055, 116.0508041, -231.8200836, 122.3880920, -341.9406128, 347.8708801
4: -202.1242828, 154.3138275, -213.3305817, 162.9325409, -365.0567017, 367.6442871
5: -180.1262665, 139.9138031, -190.1817322, 147.6407776, -327.7670288, 330.0955200
6: -173.0713806, 166.6270142, -182.6930237, 175.8118439, -348.8831787, 349.3199768
7: -188.1746674, 157.8889160, -198.5995636, 166.6114960, -354.7861633, 356.4884644
8: -227.2532196, 155.7416382, -239.7742157, 164.4581757, -391.7113647, 395.5158691
9: -171.4806519, 168.8979797, -180.9350739, 178.2280121, -349.7086182, 349.8330383

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1387823, upper bound: 372.1248071
time: 8.94 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of IS_B2_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0806562, upper bound: 372.1212662
time: 8.34 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1983157, upper bound: 372.1986076
time: 7.41 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1929428, upper bound: 372.1968414
time: 6.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 53.08 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.2263653, upper bound: 372.2230199
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.2263653, upper bound: 372.2230199
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.2197819, upper bound: 372.2195237
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.2197819, upper bound: 372.2195237
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.2225345, upper bound: 372.2146866
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.2225345, upper bound: 372.2146866
IS_B2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.1983157, upper bound: 372.1986076
IS_B2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 53.08
Output dim: 2, lower bound: -372.1929428, upper bound: 372.1968414

## BFS IS instance: IS_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -192.1798248, 152.7361908, -188.1694489, 149.6298676, -341.8096924, 340.9056396
1: -161.3049622, 135.3721008, -157.8686676, 132.5978699, -293.9028015, 293.2407532
2: -211.9992981, 137.3819427, -207.5942078, 134.6568604, -346.6561584, 344.9761353
3: -224.7154388, 118.7596207, -220.0727844, 116.2802658, -340.9956970, 338.8323364
4: -206.8281708, 157.9591522, -202.5443420, 154.7434998, -361.5716248, 360.5034790
5: -184.3735962, 143.1657410, -180.5824432, 140.2569885, -324.6305847, 323.7481689
6: -177.1427460, 170.5166016, -173.4409637, 166.9221344, -344.0647888, 343.9575500
7: -192.5531616, 161.5729370, -188.6085815, 158.2368011, -350.7899475, 350.1814880
8: -232.5876770, 159.4517212, -227.6616516, 156.1918030, -388.7794800, 387.1133728
9: -175.4546967, 172.8183899, -171.8278351, 169.2462158, -344.7008362, 344.6462097

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of IS_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1378761, upper bound: 372.1502319
time: 8.23 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of IS_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2068111, upper bound: 372.1963560
time: 8.35 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2018717, upper bound: 372.1940714
time: 8.27 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -193.6502991, 153.9278564, -188.1694489, 149.6298676, -343.2801208, 342.0972595
1: -162.4503937, 136.3641357, -157.8686676, 132.5978699, -295.0482178, 294.2327881
2: -213.6165009, 138.3427734, -207.5942078, 134.6568604, -348.2732849, 345.9369507
3: -226.3940277, 119.5676804, -220.0727844, 116.2802658, -342.6742859, 339.6404114
4: -208.4473877, 159.0875549, -202.5443420, 154.7434998, -363.1908264, 361.6318970
5: -185.7196350, 144.1782379, -180.5824432, 140.2569885, -325.9765930, 324.7606812
6: -178.4760132, 171.8209229, -173.4409637, 166.9221344, -345.3981018, 345.2618103
7: -193.9560547, 162.7417450, -188.6085815, 158.2368011, -352.1928406, 351.3503418
8: -234.3568878, 160.5946503, -227.6616516, 156.1918030, -390.5487061, 388.2562866
9: -176.7731476, 174.1392975, -171.8278351, 169.2462158, -346.0192566, 345.9670715

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of IS_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of IS_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1378761, upper bound: 372.1502319
time: 9.41 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2068111, upper bound: 372.1963560
time: 6.74 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2018717, upper bound: 372.1940714
time: 8.39 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -188.3628845, 149.7036285, -180.8238831, 143.7690582, -332.1319580, 330.5274658
1: -158.0774231, 132.6820068, -151.5677948, 127.3773727, -285.4547729, 284.2498169
2: -207.7958374, 134.6402435, -199.4961395, 129.2515869, -337.0474243, 334.1363220
3: -220.2264099, 116.3988724, -211.4344482, 111.6564026, -331.8827820, 327.8333130
4: -202.7304993, 154.8029938, -194.6394196, 148.5626373, -351.2931519, 349.4424133
5: -180.6888123, 140.3260803, -173.4427185, 134.7452240, -315.4340210, 313.7687988
6: -173.6103668, 167.1375732, -166.5958862, 160.4036865, -334.0139771, 333.7334595
7: -188.7269592, 158.3621368, -181.1964569, 152.0325012, -340.7594604, 339.5585938
8: -227.9705963, 156.2628021, -218.7143555, 149.8806458, -377.8512573, 374.9771729
9: -171.9885254, 169.4035034, -165.1796265, 162.6873016, -334.6758423, 334.5831299

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of IS_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0777389, upper bound: 372.1098790
time: 8.13 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1996787, upper bound: 372.2022325
time: 9.01 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1944753, upper bound: 372.1910133
time: 8.09 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -190.0601501, 151.0735779, -180.8238831, 143.7690582, -333.8291931, 331.8973999
1: -159.4135437, 133.8322296, -151.5677948, 127.3773727, -286.7908630, 285.4000244
2: -209.6613464, 135.7635956, -199.4961395, 129.2515869, -338.9129333, 335.2597351
3: -222.1720581, 117.3468781, -211.4344482, 111.6564026, -333.8283691, 328.7813110
4: -204.5908203, 156.1158295, -194.6394196, 148.5626373, -353.1534424, 350.7552185
5: -182.2522583, 141.5070190, -173.4427185, 134.7452240, -316.9974976, 314.9497375
6: -175.1515045, 168.6411743, -166.5958862, 160.4036865, -335.5550842, 335.2370605
7: -190.3572540, 159.7230682, -181.1964569, 152.0325012, -342.3897705, 340.9195251
8: -230.0124512, 157.5899200, -218.7143555, 149.8806458, -379.8930969, 376.3042603
9: -173.5116425, 170.9266663, -165.1796265, 162.6873016, -336.1989441, 336.1062927

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of IS_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0777389, upper bound: 372.1098790
time: 6.51 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=375.07427978515625
rel_dist={2: [-372.26984149889176, 372.2698414979088]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2054819, upper bound: 372.2175312
time: 8.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1983867, upper bound: 372.1983867
time: 7.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.95 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 16.95
Output dim: 2, lower bound: -372.2054819, upper bound: 372.2175312
IS_A2, status: Status.VERIFIED, split count: 1, time: 16.95
Output dim: 2, lower bound: -372.1983867, upper bound: 372.1983867
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=375.07427978515625
rel_dist={2: [-372.2698094205705, 372.2698094230649]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2076808, upper bound: 372.2232821
time: 9.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1984560, upper bound: 372.1984560
time: 6.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.58
Output dim: 2, lower bound: -372.2076808, upper bound: 372.2232821
IS_A2, status: Status.VERIFIED, split count: 1, time: 15.58
Output dim: 2, lower bound: -372.1984560, upper bound: 372.1984560

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -204.6883240, 162.6810608, -206.2513123, 163.9280243, -368.6163330, 368.9323425
1: -171.7985687, 144.2029266, -173.1095734, 145.2978516, -317.0963440, 317.3123779
2: -225.7713318, 146.4640656, -227.4908142, 147.5834808, -373.3547668, 373.9548950
3: -239.4956207, 126.4819412, -241.3278656, 127.4458313, -366.9413757, 367.8098145
4: -220.2167358, 168.3280945, -221.8932953, 169.6110992, -389.8278198, 390.2213440
5: -196.4882355, 152.5526733, -198.0002136, 153.7202454, -350.2084961, 350.5528564
6: -188.6708679, 181.5331116, -190.1141815, 182.9126740, -371.5835266, 371.6472778
7: -205.2094574, 172.1515198, -206.7764435, 173.4635925, -378.6729431, 378.9279785
8: -247.5364075, 169.8146362, -249.4098511, 171.1091919, -418.6455383, 419.2244873
9: -186.8628693, 184.0180969, -188.2842560, 185.4169464, -372.2798157, 372.3023682

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1800119, upper bound: 372.1870008
time: 8.23 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1826397, upper bound: 372.1907773
time: 8.30 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2040055, upper bound: 372.2193998
time: 7.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 44.62 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 44.62
Output dim: 2, lower bound: -372.1826397, upper bound: 372.1907773
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 44.62
Output dim: 2, lower bound: -372.2040055, upper bound: 372.2193998

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -202.6154480, 161.0222778, -206.2513123, 163.9280243, -366.5434570, 367.2735596
1: -170.0830688, 142.7372437, -173.1095734, 145.2978516, -315.3809204, 315.8467407
2: -223.4902191, 144.9512024, -227.4908142, 147.5834808, -371.0737000, 372.4419861
3: -237.0319214, 125.2226944, -241.3278656, 127.4458313, -364.4777222, 366.5505676
4: -217.9919128, 166.5982056, -221.8932953, 169.6110992, -387.6030273, 388.4915161
5: -194.4669342, 150.9934540, -198.0002136, 153.7202454, -348.1871948, 348.9936218
6: -186.7637177, 179.7197571, -190.1141815, 182.9126740, -369.6763916, 369.8339233
7: -203.1104279, 170.4019012, -206.7764435, 173.4635925, -376.5740051, 377.1783447
8: -245.0851593, 168.1021881, -249.4098511, 171.1091919, -416.1943054, 417.5120239
9: -184.9772034, 182.1701355, -188.2842560, 185.4169464, -370.3941040, 370.4543762

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1755299, upper bound: 372.1823824
time: 8.76 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 201

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 201

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1695674, upper bound: 372.1740877
time: 9.06 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1764165, upper bound: 372.1829091
time: 7.43 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1437634, upper bound: 372.1628827
time: 7.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 72.24 seconds
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 72.24
Output dim: 2, lower bound: -372.1764165, upper bound: 372.1829091
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 72.24
Output dim: 2, lower bound: -372.1437634, upper bound: 372.1628827
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=375.07427978515625
rel_dist={2: [-372.269821027398, 372.26982102765874]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2577557, upper bound: 372.2594363
time: 8.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2560755, upper bound: 372.2560755
time: 8.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.41
Output dim: 2, lower bound: -372.2577557, upper bound: 372.2594363
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.41
Output dim: 2, lower bound: -372.2560755, upper bound: 372.2560755

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -198.4893188, 157.8231659, -206.2513123, 163.9280243, -362.4172668, 364.0744629
1: -166.5716248, 139.8642120, -173.1095734, 145.2978516, -311.8694763, 312.9736938
2: -218.9606934, 142.0608978, -227.4908142, 147.5834808, -366.5441895, 369.5516968
3: -232.2102203, 122.6454620, -241.3278656, 127.4458313, -359.6559753, 363.9733276
4: -213.5940399, 163.2548523, -221.8932953, 169.6110992, -383.2051392, 385.1481323
5: -190.5393372, 147.9361877, -198.0002136, 153.7202454, -344.2595825, 345.9364014
6: -182.9729309, 176.0473022, -190.1141815, 182.9126740, -365.8856201, 366.1614990
7: -198.9613647, 166.9206238, -206.7764435, 173.4635925, -372.4249268, 373.6970825
8: -240.0978241, 164.7572479, -249.4098511, 171.1091919, -411.2069397, 414.1670837
9: -181.2092133, 178.4702759, -188.2842560, 185.4169464, -366.6261597, 366.7545166

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2560755, upper bound: 372.2560755
time: 8.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2560755, upper bound: 372.2560755
time: 8.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -199.2478333, 158.4422455, -202.6171112, 161.0624084, -360.3102417, 361.0592957
1: -167.1326599, 140.3604584, -170.0494995, 142.7468109, -309.8793945, 310.4099731
2: -219.7879791, 142.5113831, -223.4916534, 144.9876251, -364.7756042, 366.0029907
3: -233.0491180, 123.0219116, -237.0515747, 125.1951370, -358.2442627, 360.0733948
4: -214.4481812, 163.7952576, -218.0020752, 166.6265869, -381.0747681, 381.7973328
5: -191.1925201, 148.4223480, -194.5009918, 151.0089264, -342.2014465, 342.9233398
6: -183.6501465, 176.7235718, -186.7711029, 179.6959229, -363.3460388, 363.4946289
7: -199.6473389, 167.4870148, -203.1130066, 170.3988190, -370.0461426, 370.6000061
8: -241.0108185, 165.3170319, -245.0451813, 168.1236877, -409.1344910, 410.3622131
9: -181.8820496, 179.1553345, -184.9735565, 182.1625671, -364.0445862, 364.1289062

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1745045, upper bound: 372.1445030
time: 7.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1255711, upper bound: 372.1255711
time: 6.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.75
Output dim: 2, lower bound: -372.2560755, upper bound: 372.2560755
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.75
Output dim: 2, lower bound: -372.2560755, upper bound: 372.2560755
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 19.75
Output dim: 2, lower bound: -372.1745045, upper bound: 372.1445030
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 19.75
Output dim: 2, lower bound: -372.1255711, upper bound: 372.1255711

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -198.4893188, 157.8231659, -198.4893188, 157.8231659, -356.3125000, 356.3125000
1: -166.5716248, 139.8642120, -166.5716248, 139.8642120, -306.4357910, 306.4357910
2: -218.9606934, 142.0608978, -218.9606934, 142.0608978, -361.0216064, 361.0216064
3: -232.2102203, 122.6454620, -232.2102203, 122.6454620, -354.8556213, 354.8556213
4: -213.5940399, 163.2548523, -213.5940399, 163.2548523, -376.8488770, 376.8488770
5: -190.5393372, 147.9361877, -190.5393372, 147.9361877, -338.4755249, 338.4755249
6: -182.9729309, 176.0473022, -182.9729309, 176.0473022, -359.0202332, 359.0202332
7: -198.9613647, 166.9206238, -198.9613647, 166.9206238, -365.8819580, 365.8819580
8: -240.0978241, 164.7572479, -240.0978241, 164.7572479, -404.8550415, 404.8550415
9: -181.2092133, 178.4702759, -181.2092133, 178.4702759, -359.6795044, 359.6795044

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1968922, upper bound: 372.1861339
time: 8.35 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1470069, upper bound: 372.1792326
time: 7.88 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1349981, upper bound: 372.1508881
time: 5.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -198.4893188, 157.8231659, -199.2478333, 158.4422455, -356.9314880, 357.0709839
1: -166.5716248, 139.8642120, -167.1326599, 140.3604584, -306.9320679, 306.9967957
2: -218.9606934, 142.0608978, -219.7879791, 142.5113831, -361.4720764, 361.8488770
3: -232.2102203, 122.6454620, -233.0491180, 123.0219116, -355.2320862, 355.6945801
4: -213.5940399, 163.2548523, -214.4481812, 163.7952576, -377.3892822, 377.7030334
5: -190.5393372, 147.9361877, -191.1925201, 148.4223480, -338.9616699, 339.1287231
6: -182.9729309, 176.0473022, -183.6501465, 176.7235718, -359.6964722, 359.6974182
7: -198.9613647, 166.9206238, -199.6473389, 167.4870148, -366.4483643, 366.5679626
8: -240.0978241, 164.7572479, -241.0108185, 165.3170319, -405.4148560, 405.7680359
9: -181.2092133, 178.4702759, -181.8820496, 179.1553345, -360.3645630, 360.3523254

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 104

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1968922, upper bound: 372.1861339
time: 8.29 seconds

## Relational analysis of IS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1470069, upper bound: 372.1792326
time: 7.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1349981, upper bound: 372.1508881
time: 7.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.16 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.16
Output dim: 2, lower bound: -372.1470069, upper bound: 372.1792326
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 30.16
Output dim: 2, lower bound: -372.1349981, upper bound: 372.1508881
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 30.16
Output dim: 2, lower bound: -372.1470069, upper bound: 372.1792326
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 30.16
Output dim: 2, lower bound: -372.1349981, upper bound: 372.1508881
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=375.07427978515625
rel_dist={2: [-372.26983184842027, 372.26983184842027]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 905.61 seconds
