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
execution time: IAR + LP analysis = 1.04 + 11.00 = 12.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -372.2698839, upper bound: 372.2698839


# Binary Search by BASE starts (time budget: 2687.97 seconds, max iter: 100)

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
Binary search time: 46.83 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2641.14 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2560994, upper bound: 372.2560993
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2560994, upper bound: 372.2560993
time: 7.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.88
Output dim: 2, lower bound: -372.2560994, upper bound: 372.2560993
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.88
Output dim: 2, lower bound: -372.2560994, upper bound: 372.2560993

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523098, upper bound: 372.2523194
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523194, upper bound: 372.2523098
time: 7.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523098, upper bound: 372.2523194
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523194, upper bound: 372.2523098
time: 7.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 2, lower bound: -372.2523098, upper bound: 372.2523194
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 2, lower bound: -372.2523194, upper bound: 372.2523098
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 2, lower bound: -372.2523098, upper bound: 372.2523194
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 2, lower bound: -372.2523194, upper bound: 372.2523098

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
time: 7.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
time: 7.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
time: 8.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
time: 7.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378052, upper bound: 372.2378143
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.29
Output dim: 2, lower bound: -372.2378143, upper bound: 372.2378052

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 7.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 7.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 9.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 9.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 9.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 9.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 6.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 8.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
time: 8.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192323, upper bound: 372.2192426
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2192426, upper bound: 372.2192323

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 7.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
time: 8.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
time: 6.82 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946669, upper bound: 372.1946768
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.19
Output dim: 2, lower bound: -372.1946768, upper bound: 372.1946669
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=375.07427978515625
rel_dist={2: [-372.26984149889176, 372.2698414979088]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2561573, upper bound: 372.2561573
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2561573, upper bound: 372.2561573
time: 7.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.86
Output dim: 2, lower bound: -372.2561573, upper bound: 372.2561573
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.86
Output dim: 2, lower bound: -372.2561573, upper bound: 372.2561573

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523452, upper bound: 372.2523639
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523639, upper bound: 372.2523452
time: 8.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523452, upper bound: 372.2523639
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523639, upper bound: 372.2523452
time: 7.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.42
Output dim: 2, lower bound: -372.2523452, upper bound: 372.2523639
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.42
Output dim: 2, lower bound: -372.2523639, upper bound: 372.2523452
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.42
Output dim: 2, lower bound: -372.2523452, upper bound: 372.2523639
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.42
Output dim: 2, lower bound: -372.2523639, upper bound: 372.2523452

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
time: 6.45 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378476, upper bound: 372.2378599
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.28
Output dim: 2, lower bound: -372.2378599, upper bound: 372.2378476

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 7.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 7.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 7.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
time: 6.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
time: 6.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192641, upper bound: 372.2192820
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.19
Output dim: 2, lower bound: -372.2192820, upper bound: 372.2192641

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 7.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 7.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 7.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 7.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
time: 6.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947044, upper bound: 372.1947179
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.24
Output dim: 2, lower bound: -372.1947179, upper bound: 372.1947044
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=375.07427978515625
rel_dist={2: [-372.2698677151576, 372.26986771514873]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2561889, upper bound: 372.2561889
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2561889, upper bound: 372.2561889
time: 7.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.78
Output dim: 2, lower bound: -372.2561889, upper bound: 372.2561889
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.78
Output dim: 2, lower bound: -372.2561889, upper bound: 372.2561889

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523649, upper bound: 372.2523883
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523883, upper bound: 372.2523649
time: 6.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523649, upper bound: 372.2523883
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523883, upper bound: 372.2523649
time: 7.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.72 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.72
Output dim: 2, lower bound: -372.2523649, upper bound: 372.2523883
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.72
Output dim: 2, lower bound: -372.2523883, upper bound: 372.2523649
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.72
Output dim: 2, lower bound: -372.2523649, upper bound: 372.2523883
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.72
Output dim: 2, lower bound: -372.2523883, upper bound: 372.2523649

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
time: 6.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
time: 7.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
time: 7.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378739, upper bound: 372.2378892
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 2, lower bound: -372.2378892, upper bound: 372.2378739

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 7.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 6.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
time: 6.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2192846, upper bound: 372.2193055
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.22
Output dim: 2, lower bound: -372.2193055, upper bound: 372.2192846

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 8.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 6.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 8.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
time: 7.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
time: 7.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947256, upper bound: 372.1947396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.02
Output dim: 2, lower bound: -372.1947396, upper bound: 372.1947256
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=375.07427978515625
rel_dist={2: [-372.2698792112838, 372.2698792125061]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2562037, upper bound: 372.2562037
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2562037, upper bound: 372.2562037
time: 6.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.51
Output dim: 2, lower bound: -372.2562037, upper bound: 372.2562037
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.51
Output dim: 2, lower bound: -372.2562037, upper bound: 372.2562037

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523741, upper bound: 372.2523993
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523993, upper bound: 372.2523741
time: 5.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523741, upper bound: 372.2523993
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2523993, upper bound: 372.2523741
time: 6.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 2, lower bound: -372.2523741, upper bound: 372.2523993
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 2, lower bound: -372.2523993, upper bound: 372.2523741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 2, lower bound: -372.2523741, upper bound: 372.2523993
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 2, lower bound: -372.2523993, upper bound: 372.2523741

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
time: 5.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
time: 6.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2378864, upper bound: 372.2379033
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.17
Output dim: 2, lower bound: -372.2379033, upper bound: 372.2378864

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
time: 7.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
time: 6.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
time: 7.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
time: 6.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2192948, upper bound: 372.2193165
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192947
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.49
Output dim: 2, lower bound: -372.2193165, upper bound: 372.2192948

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 8.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -206.2513123, 163.9280243, -206.2513123, 163.9280243, -370.1793213, 370.1793213
1: -173.1095734, 145.2978516, -173.1095734, 145.2978516, -318.4073792, 318.4073792
2: -227.4908142, 147.5834808, -227.4908142, 147.5834808, -375.0742798, 375.0742798
3: -241.3278656, 127.4458313, -241.3278656, 127.4458313, -368.7736816, 368.7736816
4: -221.8932953, 169.6110992, -221.8932953, 169.6110992, -391.5043640, 391.5043640
5: -198.0002136, 153.7202454, -198.0002136, 153.7202454, -351.7204285, 351.7204285
6: -190.1141815, 182.9126740, -190.1141815, 182.9126740, -373.0268555, 373.0268555
7: -206.7764435, 173.4635925, -206.7764435, 173.4635925, -380.2400513, 380.2400513
8: -249.4098511, 171.1091919, -249.4098511, 171.1091919, -420.5189819, 420.5189819
9: -188.2842560, 185.4169464, -188.2842560, 185.4169464, -373.7012024, 373.7012024

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
time: 6.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947354, upper bound: 372.1947503
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.72
Output dim: 2, lower bound: -372.1947503, upper bound: 372.1947354
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=375.07427978515625
rel_dist={2: [-372.2698838770932, 372.2698838770932]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 1942.89 seconds
