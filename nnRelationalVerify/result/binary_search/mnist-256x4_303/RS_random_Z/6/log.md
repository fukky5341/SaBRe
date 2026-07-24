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
execution time: IAR + LP analysis = 1.05 + 11.31 = 12.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -372.2698839, upper bound: 372.2698839


# Binary Search by BASE starts (time budget: 2687.64 seconds, max iter: 100)

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
Binary search time: 46.76 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2640.88 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 88

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644976
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644976
time: 9.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.47
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644976
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.47
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644976

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

Time for backsubstitution: 0.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644922
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644922, upper bound: 372.2644976
time: 9.76 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2608843, upper bound: 372.2608843
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2608843, upper bound: 372.2608843
time: 8.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644922
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 2, lower bound: -372.2644922, upper bound: 372.2644976
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 2, lower bound: -372.2608843, upper bound: 372.2608843
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 2, lower bound: -372.2608843, upper bound: 372.2608843

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

Time for backsubstitution: 0.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644968, upper bound: 372.2644922
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644910
time: 8.04 seconds

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
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644910, upper bound: 372.2644976
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644922, upper bound: 372.2644968
time: 8.74 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2587753, upper bound: 372.2587737
time: 10.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2587737, upper bound: 372.2587753
time: 7.89 seconds

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2538068
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2538068
time: 8.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2644968, upper bound: 372.2644922
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2644976, upper bound: 372.2644910
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2644910, upper bound: 372.2644976
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2644922, upper bound: 372.2644968
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2587753, upper bound: 372.2587737
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2587737, upper bound: 372.2587753
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2538068
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.87
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2538068

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

Time for backsubstitution: 0.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2641010, upper bound: 372.2641013
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2641009, upper bound: 372.2641014
time: 8.34 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2432220, upper bound: 372.2432250
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2432220, upper bound: 372.2432250
time: 8.09 seconds

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
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2217368, upper bound: 372.2217214
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2217368, upper bound: 372.2217214
time: 6.56 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644922, upper bound: 372.2644937
time: 10.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2644904, upper bound: 372.2644968
time: 10.44 seconds

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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1761848, upper bound: 372.1761887
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1761848, upper bound: 372.1761887
time: 6.15 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2510925, upper bound: 372.2510999
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2510925, upper bound: 372.2510999
time: 8.37 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2537752, upper bound: 372.2538068
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2537752
time: 8.02 seconds

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2537899
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2537899, upper bound: 372.2538068
time: 8.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2641010, upper bound: 372.2641013
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2641009, upper bound: 372.2641014
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2432220, upper bound: 372.2432250
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2432220, upper bound: 372.2432250
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2217368, upper bound: 372.2217214
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2217368, upper bound: 372.2217214
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2644922, upper bound: 372.2644937
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2644904, upper bound: 372.2644968
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.1761848, upper bound: 372.1761887
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.1761848, upper bound: 372.1761887
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2510925, upper bound: 372.2510999
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2510925, upper bound: 372.2510999
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2537752, upper bound: 372.2538068
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2537752
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2538068, upper bound: 372.2537899
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.41
Output dim: 2, lower bound: -372.2537899, upper bound: 372.2538068

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1349228, upper bound: 372.1349570
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1349228, upper bound: 372.1349570
time: 6.78 seconds

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

Time for backsubstitution: 0.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2190164, upper bound: 372.2190145
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2190164, upper bound: 372.2190145
time: 7.91 seconds

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0870648, upper bound: 372.0870689
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0870648, upper bound: 372.0870689
time: 5.95 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2432216, upper bound: 372.2432250
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2432220, upper bound: 372.2432243
time: 7.36 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2111713, upper bound: 372.2111453
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2111500, upper bound: 372.2111617
time: 6.53 seconds

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
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2217368, upper bound: 372.2217157
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2217244, upper bound: 372.2217214
time: 7.87 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2618552, upper bound: 372.2618577
time: 8.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2618552, upper bound: 372.2618577
time: 8.79 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1718630, upper bound: 372.1718546
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1718630, upper bound: 372.1718546
time: 6.57 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2023443, upper bound: 372.2023445
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2023443, upper bound: 372.2023445
time: 6.95 seconds

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946322, upper bound: 372.1946418
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1946322, upper bound: 372.1946418
time: 6.64 seconds

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

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2537752, upper bound: 372.2537268
time: 8.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2537198, upper bound: 372.2538068
time: 8.34 seconds

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

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1910728, upper bound: 372.1910723
time: 9.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1910728, upper bound: 372.1910723
time: 9.04 seconds

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2368976, upper bound: 372.2368891
time: 9.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2368976, upper bound: 372.2368891
time: 8.42 seconds

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2513164, upper bound: 372.2513422
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2513410, upper bound: 372.2513206
time: 9.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1349228, upper bound: 372.1349570
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1349228, upper bound: 372.1349570
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2190164, upper bound: 372.2190145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2190164, upper bound: 372.2190145
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.0870648, upper bound: 372.0870689
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.0870648, upper bound: 372.0870689
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2432216, upper bound: 372.2432250
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2432220, upper bound: 372.2432243
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2111713, upper bound: 372.2111453
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2111500, upper bound: 372.2111617
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2217368, upper bound: 372.2217157
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2217244, upper bound: 372.2217214
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2618552, upper bound: 372.2618577
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2618552, upper bound: 372.2618577
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1718630, upper bound: 372.1718546
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1718630, upper bound: 372.1718546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2023443, upper bound: 372.2023445
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2023443, upper bound: 372.2023445
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1946322, upper bound: 372.1946418
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1946322, upper bound: 372.1946418
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2537752, upper bound: 372.2537268
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2537198, upper bound: 372.2538068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1910728, upper bound: 372.1910723
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.1910728, upper bound: 372.1910723
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2368976, upper bound: 372.2368891
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2368976, upper bound: 372.2368891
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2513164, upper bound: 372.2513422
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.75
Output dim: 2, lower bound: -372.2513410, upper bound: 372.2513206

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2190045, upper bound: 372.2190145
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2190164, upper bound: 372.2190001
time: 8.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2131236, upper bound: 372.2131425
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2131239, upper bound: 372.2131394
time: 8.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2132322, upper bound: 372.2132545
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2132322, upper bound: 372.2132545
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2414512, upper bound: 372.2414339
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2414441, upper bound: 372.2414347
time: 7.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1644985, upper bound: 372.1645044
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1644985, upper bound: 372.1645044
time: 5.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1415305, upper bound: 372.1415305
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1415305, upper bound: 372.1415305
time: 6.22 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2190045, upper bound: 372.2190145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2190164, upper bound: 372.2190001
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2131236, upper bound: 372.2131425
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2131239, upper bound: 372.2131394
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2132322, upper bound: 372.2132545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2132322, upper bound: 372.2132545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2414512, upper bound: 372.2414339
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.2414441, upper bound: 372.2414347
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.1644985, upper bound: 372.1645044
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.1644985, upper bound: 372.1645044
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.1415305, upper bound: 372.1415305
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.75
Output dim: 2, lower bound: -372.1415305, upper bound: 372.1415305
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2618552, upper bound: 372.2618577
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2618552, upper bound: 372.2618577
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2537752, upper bound: 372.2537268
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2537198, upper bound: 372.2538068
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2368976, upper bound: 372.2368891
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2368976, upper bound: 372.2368891
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2513164, upper bound: 372.2513422
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.75
Output dim: 2, lower bound: -372.2513410, upper bound: 372.2513206
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=375.07427978515625
rel_dist={2: [-372.26984149889176, 372.2698414979088]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2614874, upper bound: 372.2614874
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2614874, upper bound: 372.2614874
time: 8.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.81
Output dim: 2, lower bound: -372.2614874, upper bound: 372.2614874
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.81
Output dim: 2, lower bound: -372.2614874, upper bound: 372.2614874

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2614821, upper bound: 372.2614855
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2614855, upper bound: 372.2614821
time: 7.91 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2302909, upper bound: 372.2302909
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2302909, upper bound: 372.2302909
time: 7.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.04
Output dim: 2, lower bound: -372.2614821, upper bound: 372.2614855
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.04
Output dim: 2, lower bound: -372.2614855, upper bound: 372.2614821
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.04
Output dim: 2, lower bound: -372.2302909, upper bound: 372.2302909
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.04
Output dim: 2, lower bound: -372.2302909, upper bound: 372.2302909

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
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2527344, upper bound: 372.2527403
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2527344, upper bound: 372.2527403
time: 8.88 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2519197, upper bound: 372.2519174
time: 9.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2519197, upper bound: 372.2519174
time: 9.80 seconds

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

Time for backsubstitution: 0.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 209

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2302909, upper bound: 372.2302870
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2302870, upper bound: 372.2302909
time: 6.89 seconds

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
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2258535, upper bound: 372.2258536
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2258536, upper bound: 372.2258535
time: 7.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2527344, upper bound: 372.2527403
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2527344, upper bound: 372.2527403
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2519197, upper bound: 372.2519174
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2519197, upper bound: 372.2519174
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2302909, upper bound: 372.2302870
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2302870, upper bound: 372.2302909
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2258535, upper bound: 372.2258536
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.63
Output dim: 2, lower bound: -372.2258536, upper bound: 372.2258535

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 157

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2493739, upper bound: 372.2493945
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2493874, upper bound: 372.2493786
time: 8.38 seconds

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

Time for backsubstitution: 0.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2468449, upper bound: 372.2468741
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2468449, upper bound: 372.2468741
time: 9.53 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2319027, upper bound: 372.2318951
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2319027, upper bound: 372.2318951
time: 8.21 seconds

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
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2028717, upper bound: 372.2028711
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2028717, upper bound: 372.2028711
time: 8.90 seconds

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2235281, upper bound: 372.2234934
time: 8.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2235186, upper bound: 372.2235049
time: 6.20 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2264561, upper bound: 372.2264703
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2264643, upper bound: 372.2264583
time: 7.28 seconds

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2094098, upper bound: 372.2094064
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2094098, upper bound: 372.2094064
time: 7.39 seconds

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2258424, upper bound: 372.2258535
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2258536, upper bound: 372.2258424
time: 9.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2493739, upper bound: 372.2493945
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2493874, upper bound: 372.2493786
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2468449, upper bound: 372.2468741
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2468449, upper bound: 372.2468741
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2319027, upper bound: 372.2318951
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2319027, upper bound: 372.2318951
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2028717, upper bound: 372.2028711
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2028717, upper bound: 372.2028711
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2235281, upper bound: 372.2234934
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2235186, upper bound: 372.2235049
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2264561, upper bound: 372.2264703
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2264643, upper bound: 372.2264583
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2094098, upper bound: 372.2094064
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2094098, upper bound: 372.2094064
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2258424, upper bound: 372.2258535
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.83
Output dim: 2, lower bound: -372.2258536, upper bound: 372.2258424

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2493739, upper bound: 372.2493921
time: 9.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2493710, upper bound: 372.2493945
time: 8.48 seconds

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

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2183518, upper bound: 372.2183733
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2183518, upper bound: 372.2183733
time: 8.15 seconds

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1975955, upper bound: 372.1975955
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1975955, upper bound: 372.1975955
time: 7.46 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2468449, upper bound: 372.2468740
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2468448, upper bound: 372.2468741
time: 9.84 seconds

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2214190, upper bound: 372.2214046
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2214190, upper bound: 372.2214046
time: 10.94 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2318686, upper bound: 372.2318655
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2318686, upper bound: 372.2318655
time: 9.26 seconds

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
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 209

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2218524, upper bound: 372.2218358
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2218524, upper bound: 372.2218358
time: 8.03 seconds

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
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1474197, upper bound: 372.1474207
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1474197, upper bound: 372.1474207
time: 7.40 seconds

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1339768, upper bound: 372.1339759
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1339768, upper bound: 372.1339759
time: 6.57 seconds

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
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 56

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1150902, upper bound: 372.1151039
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1150902, upper bound: 372.1151039
time: 6.18 seconds

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
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2243295, upper bound: 372.2243288
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2243295, upper bound: 372.2243288
time: 9.43 seconds

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
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 121

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2255767, upper bound: 372.2255694
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2255796, upper bound: 372.2255639
time: 8.09 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2493739, upper bound: 372.2493921
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2493710, upper bound: 372.2493945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2183518, upper bound: 372.2183733
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2183518, upper bound: 372.2183733
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1975955, upper bound: 372.1975955
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1975955, upper bound: 372.1975955
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2468449, upper bound: 372.2468740
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2468448, upper bound: 372.2468741
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2214190, upper bound: 372.2214046
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2214190, upper bound: 372.2214046
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2318686, upper bound: 372.2318655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2318686, upper bound: 372.2318655
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2218524, upper bound: 372.2218358
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2218524, upper bound: 372.2218358
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1474197, upper bound: 372.1474207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1474197, upper bound: 372.1474207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1339768, upper bound: 372.1339759
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1339768, upper bound: 372.1339759
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1150902, upper bound: 372.1151039
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.1150902, upper bound: 372.1151039
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2243295, upper bound: 372.2243288
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2243295, upper bound: 372.2243288
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2255767, upper bound: 372.2255694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.55
Output dim: 2, lower bound: -372.2255796, upper bound: 372.2255639

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2292433, upper bound: 372.2292554
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2292433, upper bound: 372.2292554
time: 8.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2493710, upper bound: 372.2493945
time: 9.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2493698, upper bound: 372.2493944
time: 8.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2440326, upper bound: 372.2440805
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2440326, upper bound: 372.2440805
time: 9.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1750221, upper bound: 372.1750131
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1750221, upper bound: 372.1750131
time: 7.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2194082, upper bound: 372.2194050
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2194120, upper bound: 372.2193989
time: 7.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2214190, upper bound: 372.2214036
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2214180, upper bound: 372.2214046
time: 9.70 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2292433, upper bound: 372.2292554
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2292433, upper bound: 372.2292554
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2493710, upper bound: 372.2493945
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2493698, upper bound: 372.2493944
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2440326, upper bound: 372.2440805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2440326, upper bound: 372.2440805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.1750221, upper bound: 372.1750131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.1750221, upper bound: 372.1750131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2194082, upper bound: 372.2194050
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2194120, upper bound: 372.2193989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2214190, upper bound: 372.2214036
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.25
Output dim: 2, lower bound: -372.2214180, upper bound: 372.2214046
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2318686, upper bound: 372.2318655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2318686, upper bound: 372.2318655
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2218524, upper bound: 372.2218358
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2218524, upper bound: 372.2218358
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2243295, upper bound: 372.2243288
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2243295, upper bound: 372.2243288
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2255767, upper bound: 372.2255694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.25
Output dim: 2, lower bound: -372.2255796, upper bound: 372.2255639
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=375.07427978515625
rel_dist={2: [-372.2698094205705, 372.2698094230649]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407038, upper bound: 372.2407038
time: 9.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407038, upper bound: 372.2407038
time: 9.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 19.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 19.76
Output dim: 2, lower bound: -372.2407038, upper bound: 372.2407038
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 19.76
Output dim: 2, lower bound: -372.2407038, upper bound: 372.2407038

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
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407038, upper bound: 372.2407033
time: 10.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407033, upper bound: 372.2407038
time: 11.21 seconds

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
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1883057, upper bound: 372.1883057
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1883057, upper bound: 372.1883057
time: 8.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.21
Output dim: 2, lower bound: -372.2407038, upper bound: 372.2407033
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.21
Output dim: 2, lower bound: -372.2407033, upper bound: 372.2407038
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 17.21
Output dim: 2, lower bound: -372.1883057, upper bound: 372.1883057
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 17.21
Output dim: 2, lower bound: -372.1883057, upper bound: 372.1883057

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

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2406913, upper bound: 372.2406911
time: 12.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2406913, upper bound: 372.2406911
time: 12.90 seconds

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407033, upper bound: 372.2407037
time: 9.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407033, upper bound: 372.2407038
time: 14.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.41 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.41
Output dim: 2, lower bound: -372.2406913, upper bound: 372.2406911
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.41
Output dim: 2, lower bound: -372.2406913, upper bound: 372.2406911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.41
Output dim: 2, lower bound: -372.2407033, upper bound: 372.2407037
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.41
Output dim: 2, lower bound: -372.2407033, upper bound: 372.2407038

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 74

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1944978, upper bound: 372.1944976
time: 13.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1944978, upper bound: 372.1944976
time: 14.06 seconds

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
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1202493, upper bound: 372.1202471
time: 10.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1202493, upper bound: 372.1202471
time: 10.33 seconds

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

Time for backsubstitution: 0.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1236349, upper bound: 372.1236367
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1236349, upper bound: 372.1236367
time: 7.41 seconds

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
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2400400, upper bound: 372.2400483
time: 11.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2400452, upper bound: 372.2400403
time: 12.19 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.1944978, upper bound: 372.1944976
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.1944978, upper bound: 372.1944976
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.1202493, upper bound: 372.1202471
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.1202493, upper bound: 372.1202471
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.1236349, upper bound: 372.1236367
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.1236349, upper bound: 372.1236367
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.2400400, upper bound: 372.2400483
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.83
Output dim: 2, lower bound: -372.2400452, upper bound: 372.2400403

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

Time for backsubstitution: 0.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2360153, upper bound: 372.2360154
time: 10.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2360150, upper bound: 372.2360159
time: 13.70 seconds

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
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1984135, upper bound: 372.1984016
time: 10.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1984135, upper bound: 372.1984016
time: 11.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.12 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.12
Output dim: 2, lower bound: -372.2360153, upper bound: 372.2360154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.12
Output dim: 2, lower bound: -372.2360150, upper bound: 372.2360159
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.12
Output dim: 2, lower bound: -372.1984135, upper bound: 372.1984016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.12
Output dim: 2, lower bound: -372.1984135, upper bound: 372.1984016

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2081863, upper bound: 372.2081901
time: 14.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2081863, upper bound: 372.2081901
time: 12.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 88

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2360107, upper bound: 372.2360113
time: 10.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2360107, upper bound: 372.2360113
time: 14.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.74 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.74
Output dim: 2, lower bound: -372.2081863, upper bound: 372.2081901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.74
Output dim: 2, lower bound: -372.2081863, upper bound: 372.2081901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.74
Output dim: 2, lower bound: -372.2360107, upper bound: 372.2360113
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.74
Output dim: 2, lower bound: -372.2360107, upper bound: 372.2360113

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2289463, upper bound: 372.2289475
time: 12.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2289437, upper bound: 372.2289509
time: 14.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2126887, upper bound: 372.2126918
time: 12.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2126887, upper bound: 372.2126918
time: 11.82 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.84 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -372.2289463, upper bound: 372.2289475
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -372.2289437, upper bound: 372.2289509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.84
Output dim: 2, lower bound: -372.2126887, upper bound: 372.2126918
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.84
Output dim: 2, lower bound: -372.2126887, upper bound: 372.2126918

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2289479, upper bound: 372.2289415
time: 13.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2289380, upper bound: 372.2289475
time: 10.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2150298, upper bound: 372.2150214
time: 12.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2150298, upper bound: 372.2150214
time: 14.88 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 28.09 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 28.09
Output dim: 2, lower bound: -372.2289479, upper bound: 372.2289415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 28.09
Output dim: 2, lower bound: -372.2289380, upper bound: 372.2289475
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 28.09
Output dim: 2, lower bound: -372.2150298, upper bound: 372.2150214
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 28.09
Output dim: 2, lower bound: -372.2150298, upper bound: 372.2150214

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0934755, upper bound: 372.0934761
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.0934755, upper bound: 372.0934761
time: 8.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 205

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1689175, upper bound: 372.1689155
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1689175, upper bound: 372.1689155
time: 7.21 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 15.34 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 15.34
Output dim: 2, lower bound: -372.0934755, upper bound: 372.0934761
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 15.34
Output dim: 2, lower bound: -372.0934755, upper bound: 372.0934761
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 15.34
Output dim: 2, lower bound: -372.1689175, upper bound: 372.1689155
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 15.34
Output dim: 2, lower bound: -372.1689175, upper bound: 372.1689155
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=375.07427978515625
rel_dist={2: [-372.26978283690505, 372.26978280960407]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 205
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 74
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 158
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 157
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 193
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 209
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 88
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 121
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1643346, upper bound: 372.1643346
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1643346, upper bound: 372.1643346
time: 6.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.38 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 13.38
Output dim: 2, lower bound: -372.1643346, upper bound: 372.1643346
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 13.38
Output dim: 2, lower bound: -372.1643346, upper bound: 372.1643346
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=375.07427978515625
rel_dist={2: [-372.2697968624731, 372.2697968624731]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 1708.04 seconds
