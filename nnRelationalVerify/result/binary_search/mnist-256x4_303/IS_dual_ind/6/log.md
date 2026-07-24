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
execution time: IAR + LP analysis = 1.01 + 10.84 = 11.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -372.2698839, upper bound: 372.2698839


# Binary Search by BASE starts (time budget: 2688.15 seconds, max iter: 100)

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
Binary search time: 46.67 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2641.48 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2681256, upper bound: 372.2673914
time: 7.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2687684, upper bound: 372.2687684
time: 7.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.94
Output dim: 2, lower bound: -372.2681256, upper bound: 372.2673914
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.94
Output dim: 2, lower bound: -372.2687684, upper bound: 372.2687684

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -195.3206635, 155.3128815, -205.6799011, 163.4774170, -358.7980957, 360.9927063
1: -163.8407288, 137.6345367, -172.6238403, 144.8972473, -308.7379456, 310.2583313
2: -215.4388123, 139.8561096, -226.8607941, 147.1797180, -362.6184387, 366.7169189
3: -228.5296783, 120.7476654, -240.6589203, 127.0955963, -355.6252441, 361.4065857
4: -210.1535492, 160.6972504, -221.2812500, 169.1458435, -379.2993774, 381.9784546
5: -187.5573120, 145.6229858, -197.4541779, 153.2968292, -340.8541260, 343.0771484
6: -180.0850830, 173.1997681, -189.5906219, 182.4046631, -362.4897461, 362.7904053
7: -195.8436737, 164.3047943, -206.2051697, 172.9848175, -368.8284607, 370.5098572
8: -236.2275085, 162.1352081, -248.7203064, 170.6400909, -406.8676147, 410.8555298
9: -178.2790680, 175.5395508, -187.7613220, 184.8996735, -363.1787109, 363.3008728

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2577678, upper bound: 372.2559485
time: 8.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2573309, upper bound: 372.2561650
time: 8.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -199.6509399, 158.7141571, -205.8966064, 163.6476746, -363.2986145, 364.6107788
1: -167.5093231, 140.6702576, -172.8085175, 145.0491180, -312.5583801, 313.4787292
2: -220.2118988, 142.9197845, -227.0995789, 147.3328857, -367.5447693, 370.0193481
3: -233.5867157, 123.4031067, -240.9117279, 127.2286682, -360.8153687, 364.3148193
4: -214.8302155, 164.2307281, -221.5135498, 169.3219452, -384.1521606, 385.7442322
5: -191.6817169, 148.8310699, -197.6605835, 153.4575195, -345.1392212, 346.4916077
6: -184.0677643, 177.0479431, -189.7891693, 182.5975342, -366.6652832, 366.8370972
7: -200.1786194, 167.9325256, -206.4218903, 173.1663208, -373.3449402, 374.3544312
8: -241.4618378, 165.6829529, -248.9825745, 170.8175201, -412.2793579, 414.6655273
9: -182.2528229, 179.4474182, -187.9600525, 185.0960388, -367.3488159, 367.4074707

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2590741, upper bound: 372.2585696
time: 8.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2587198, upper bound: 372.2587198
time: 8.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.96
Output dim: 2, lower bound: -372.2577678, upper bound: 372.2559485
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.96
Output dim: 2, lower bound: -372.2573309, upper bound: 372.2561650
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.96
Output dim: 2, lower bound: -372.2590741, upper bound: 372.2585696
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.96
Output dim: 2, lower bound: -372.2587198, upper bound: 372.2587198

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -195.3206635, 155.3128815, -194.2479858, 154.3994446, -349.7200928, 349.5607910
1: -163.8407288, 137.6345367, -162.9534149, 136.8051758, -300.6458435, 300.5879517
2: -215.4388123, 139.8561096, -214.2929535, 139.0237885, -354.4624939, 354.1490479
3: -228.5296783, 120.7476654, -227.2042236, 120.0596542, -348.5893250, 347.9519043
4: -210.1535492, 160.6972504, -209.1640472, 159.7240295, -369.8775635, 369.8612976
5: -187.5573120, 145.6229858, -186.4753418, 144.7980194, -332.3552856, 332.0983276
6: -180.0850830, 173.1997681, -179.0248413, 172.3119507, -352.3970337, 352.2246094
7: -195.8436737, 164.3047943, -194.8086090, 163.3911438, -359.2347717, 359.1133118
8: -236.2275085, 162.1352081, -234.9100189, 161.1900024, -397.4174805, 397.0452271
9: -178.2790680, 175.5395508, -177.3568726, 174.6519012, -352.9309692, 352.8964233

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2567678, upper bound: 372.2554087
time: 8.12 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2567678, upper bound: 372.2558322
time: 9.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -193.0870667, 153.5391846, -195.3167419, 155.2375641, -348.3246460, 348.8559265
1: -161.9461365, 136.0525055, -163.7735748, 137.5153961, -299.4615479, 299.8260803
2: -212.9789124, 138.2596741, -215.4346771, 139.7239075, -352.7028198, 353.6943054
3: -225.9004517, 119.3680267, -228.3906250, 120.6717834, -346.5722351, 347.7586060
4: -207.7807312, 158.8537903, -210.3557892, 160.5620270, -368.3426819, 369.2095337
5: -185.4130859, 143.9602814, -187.5014038, 145.5614929, -330.9745483, 331.4616699
6: -178.0215759, 171.2250061, -179.9911194, 173.2457275, -351.2672729, 351.2161255
7: -193.6117859, 162.4275513, -195.8497314, 164.2511597, -357.8629456, 358.2772522
8: -233.5238647, 160.2809906, -236.1471100, 161.9792328, -395.5031128, 396.4281006
9: -176.2424622, 173.5358887, -178.3085022, 175.5995941, -351.8420410, 351.8443604

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2411484, upper bound: 372.2420403
time: 8.04 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2389222, upper bound: 372.2375286
time: 7.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -199.6509399, 158.7141571, -194.4586487, 154.5649414, -354.2158813, 353.1727905
1: -167.5093231, 140.6702576, -163.1330872, 136.9528809, -304.4621887, 303.8033142
2: -220.2118988, 142.9197845, -214.5251617, 139.1727448, -359.3846436, 357.4448853
3: -233.5867157, 123.4031067, -227.4503174, 120.1889038, -353.7756348, 350.8534241
4: -214.8302155, 164.2307281, -209.3901520, 159.8952179, -374.7253723, 373.6207886
5: -191.6817169, 148.8310699, -186.6760254, 144.9543457, -336.6360474, 335.5070496
6: -184.0677643, 177.0479431, -179.2178345, 172.4996338, -356.5673828, 356.2657471
7: -200.1786194, 167.9325256, -195.0194702, 163.5677338, -363.7463379, 362.9519958
8: -241.4618378, 165.6829529, -235.1645966, 161.3626556, -402.8244629, 400.8475342
9: -182.2528229, 179.4474182, -177.5504761, 174.8427124, -357.0954895, 356.9978333

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2581191, upper bound: 372.2581191
time: 7.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2581191, upper bound: 372.2584332
time: 8.04 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -197.4337921, 156.9535980, -195.5383148, 155.4120026, -352.8457947, 352.4919128
1: -165.6288300, 139.0997620, -163.9626007, 137.6706696, -303.2994995, 303.0623779
2: -217.7707062, 141.3348389, -215.6790924, 139.8806763, -357.6513672, 357.0138855
3: -230.9761047, 122.0335236, -228.6496887, 120.8079224, -351.7840271, 350.6831970
4: -212.4753571, 162.4011993, -210.5933685, 160.7425232, -373.2178955, 372.9945068
5: -189.5531464, 147.1809235, -187.7122955, 145.7260284, -335.2791748, 334.8932190
6: -182.0201111, 175.0876617, -180.1945953, 173.4427948, -355.4628601, 355.2822266
7: -197.9628448, 166.0690155, -196.0714569, 164.4367065, -362.3995361, 362.1404419
8: -238.7789001, 163.8429565, -236.4158630, 162.1614838, -400.9403687, 400.2588196
9: -180.2313385, 177.4584961, -178.5118103, 175.8005371, -356.0318604, 355.9702759

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2422976, upper bound: 372.2448806
time: 8.47 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2403066, upper bound: 372.2403066
time: 7.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2567678, upper bound: 372.2554087
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2567678, upper bound: 372.2558322
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2411484, upper bound: 372.2420403
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2389222, upper bound: 372.2375286
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2581191, upper bound: 372.2581191
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2581191, upper bound: 372.2584332
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2422976, upper bound: 372.2448806
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 2, lower bound: -372.2403066, upper bound: 372.2403066

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -183.9405060, 146.2760010, -194.2479858, 154.3994446, -338.3399048, 340.5239868
1: -154.2132874, 129.5789032, -162.9534149, 136.8051758, -291.0184326, 292.5323181
2: -202.9273834, 131.7366943, -214.2929535, 139.0237885, -341.9511719, 346.0296326
3: -215.1352844, 113.7425995, -227.2042236, 120.0596542, -335.1949463, 340.9468384
4: -198.0909576, 151.3186493, -209.1640472, 159.7240295, -357.8149719, 360.4826965
5: -176.6286469, 137.1627808, -186.4753418, 144.7980194, -321.4265747, 323.6380310
6: -169.5669556, 163.1527557, -179.0248413, 172.3119507, -341.8788757, 342.1775818
7: -184.4985504, 154.7545166, -194.8086090, 163.3911438, -347.8896179, 349.5630798
8: -222.4814301, 152.7266693, -234.9100189, 161.1900024, -383.6712952, 387.6366882
9: -167.9209442, 165.3393860, -177.3568726, 174.6519012, -342.5728455, 342.6962585

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2434358, upper bound: 372.2391723
time: 7.16 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2372992
time: 7.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -184.7630005, 146.9142303, -194.2479858, 154.3994446, -339.1623535, 341.1622009
1: -154.8224792, 130.1094666, -162.9534149, 136.8051758, -291.6276550, 293.0628662
2: -203.7982635, 132.2575531, -214.2929535, 139.0237885, -342.8220215, 346.5505066
3: -216.0290222, 114.1989517, -227.2042236, 120.0596542, -336.0886841, 341.4031677
4: -199.0048676, 151.9495850, -209.1640472, 159.7240295, -358.7288513, 361.1136169
5: -177.4187622, 137.7374268, -186.4753418, 144.7980194, -322.2167969, 324.2127380
6: -170.3013763, 163.8634491, -179.0248413, 172.3119507, -342.6132812, 342.8882751
7: -185.2854462, 155.4028168, -194.8086090, 163.3911438, -348.6765747, 350.2114258
8: -223.4143219, 153.3123474, -234.9100189, 161.1900024, -384.6042786, 388.2223511
9: -168.6418762, 166.0639954, -177.3568726, 174.6519012, -343.2937622, 343.4208374

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2434358, upper bound: 372.2393725
time: 9.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
time: 7.93 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -185.3598480, 147.4630737, -195.3167419, 155.2375641, -340.5974121, 342.7797546
1: -155.4373627, 130.6425629, -163.7735748, 137.5153961, -292.9527588, 294.4161377
2: -204.4875946, 132.7601776, -215.4346771, 139.7239075, -344.2114868, 348.1948547
3: -216.8226776, 114.5881119, -228.3906250, 120.6717834, -337.4944458, 342.9786377
4: -199.5163422, 152.5262909, -210.3557892, 160.5620270, -360.0782776, 362.8820801
5: -177.9870911, 138.2009735, -187.5014038, 145.5614929, -323.5485535, 325.7023315
6: -170.9117279, 164.3899994, -179.9911194, 173.2457275, -344.1574097, 344.3811035
7: -185.8299255, 155.9126282, -195.8497314, 164.2511597, -350.0810852, 351.7622986
8: -224.2525787, 153.9574738, -236.1471100, 161.9792328, -386.2318115, 390.1045837
9: -169.1975555, 166.6198578, -178.3085022, 175.5995941, -344.7971497, 344.9283447

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2389222, upper bound: 372.2375286
time: 9.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2389222, upper bound: 372.2375286
time: 8.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -186.1193085, 148.0758820, -192.7116394, 153.1837769, -339.3031006, 340.7875061
1: -155.9946136, 131.1351776, -161.5805359, 135.6870728, -291.6817017, 292.7156982
2: -205.3137207, 133.2117310, -212.5686035, 137.8632965, -343.1770020, 345.7803345
3: -217.6546326, 114.9604874, -225.3260498, 119.0583725, -336.7130127, 340.2865295
4: -200.3634796, 153.0643311, -207.5662842, 158.4227905, -358.7861633, 360.6306152
5: -178.6352539, 138.6867676, -184.9932251, 143.6182404, -322.2534485, 323.6799622
6: -171.5843506, 165.0631409, -177.5943451, 170.9406738, -342.5249939, 342.6574402
7: -186.5145416, 156.4786835, -193.2243195, 162.0546570, -348.5692139, 349.7030029
8: -225.1608582, 154.5166016, -233.0186005, 159.8391266, -385.0000000, 387.5351868
9: -169.8682098, 167.3036652, -175.9355164, 173.2676086, -343.1358032, 343.2391357

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2370422, upper bound: 372.2370422
time: 7.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2370422, upper bound: 372.2375286
time: 7.83 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -188.1866608, 149.6107635, -194.4586487, 154.5649414, -342.7515564, 344.0693970
1: -157.8115234, 132.5553894, -163.1330872, 136.9528809, -294.7644043, 295.6884460
2: -207.6076660, 134.7412567, -214.5251617, 139.1727448, -346.7803955, 349.2664185
3: -220.0957794, 116.3474121, -227.4503174, 120.1889038, -340.2846680, 343.7977295
4: -202.6794434, 154.7826233, -209.3901520, 159.8952179, -362.5746155, 364.1727295
5: -180.6741180, 140.3087006, -186.6760254, 144.9543457, -325.6284790, 326.9846497
6: -173.4722290, 166.9274902, -179.2178345, 172.4996338, -345.9718628, 346.1453247
7: -188.7507629, 158.3126831, -195.0194702, 163.5677338, -352.3184814, 353.3320923
8: -227.6111908, 156.2059021, -235.1645966, 161.3626556, -388.9738464, 391.3704529
9: -171.8202515, 169.1698303, -177.5504761, 174.8427124, -346.6629639, 346.7202454

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2449358, upper bound: 372.2419277
time: 9.09 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2399987
time: 7.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -189.1215210, 150.3420410, -194.4586487, 154.5649414, -343.6864014, 344.8006897
1: -158.5181427, 133.1634979, -163.1330872, 136.9528809, -295.4710083, 296.2965698
2: -208.6014252, 135.3455353, -214.5251617, 139.1727448, -347.7741699, 349.8706970
3: -221.1181030, 116.8704834, -227.4503174, 120.1889038, -341.3070068, 344.3208008
4: -203.7108917, 155.5090027, -209.3901520, 159.8952179, -363.6061096, 364.8991394
5: -181.5682373, 140.9720917, -186.6760254, 144.9543457, -326.5225830, 327.6480713
6: -174.3113403, 167.7360535, -179.2178345, 172.4996338, -346.8109436, 346.9538879
7: -189.6472168, 159.0542145, -195.0194702, 163.5677338, -353.2149048, 354.0736694
8: -228.6891785, 156.8935089, -235.1645966, 161.3626556, -390.0518188, 392.0581055
9: -172.6422882, 170.0003510, -177.5504761, 174.8427124, -347.4849854, 347.5507507

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2449358, upper bound: 372.2422569
time: 6.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
time: 8.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -189.6658783, 150.8441162, -195.5383148, 155.4120026, -345.0778809, 346.3824158
1: -159.0864868, 133.6622772, -163.9626007, 137.6706696, -296.7571411, 297.6248779
2: -209.2348633, 135.8092041, -215.6790924, 139.8806763, -349.1155396, 351.4882507
3: -221.8514862, 117.2290649, -228.6496887, 120.8079224, -342.6594238, 345.8787537
4: -204.1699524, 156.0412292, -210.5933685, 160.7425232, -364.9124451, 366.6345520
5: -182.0859833, 141.3945770, -187.7122955, 145.7260284, -327.8120117, 329.1068726
6: -174.8741913, 168.2174225, -180.1945953, 173.4427948, -348.3169861, 348.4120178
7: -190.1438446, 159.5215149, -196.0714569, 164.4367065, -354.5805359, 355.5929260
8: -229.4601746, 157.4869385, -236.4158630, 162.1614838, -391.6216431, 393.9028015
9: -173.1528625, 170.5063782, -178.5118103, 175.8005371, -348.9533997, 349.0181580

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2403066, upper bound: 372.2403066
time: 7.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2403066, upper bound: 372.2403066
time: 7.43 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -190.2883759, 151.3569641, -192.9338837, 153.3586884, -343.6470642, 344.2908325
1: -159.5312195, 134.0548706, -161.7701111, 135.8427734, -295.3739929, 295.8249817
2: -209.9085236, 136.1610870, -212.8135834, 138.0204163, -347.9289246, 348.9746399
3: -222.5278625, 117.5141068, -225.5858765, 119.1948700, -341.7227173, 343.0999756
4: -204.8663483, 156.4666290, -207.8045654, 158.6037445, -363.4700623, 364.2711792
5: -182.6097565, 141.7781830, -185.2048340, 143.7831726, -326.3929443, 326.9830017
6: -175.4176483, 168.7714081, -177.7983551, 171.1383057, -346.5559082, 346.5697632
7: -190.6843567, 159.9704742, -193.4465332, 162.2408600, -352.9252014, 353.4169922
8: -230.2047882, 157.9357147, -233.2882080, 160.0216827, -390.2264404, 391.2239380
9: -173.6960907, 171.0709381, -176.1394196, 173.4690552, -347.1651001, 347.2103577

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2375286, upper bound: 372.2389222
time: 9.13 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2375286, upper bound: 372.2403066
time: 8.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.19 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2434358, upper bound: 372.2391723
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2372992
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2434358, upper bound: 372.2393725
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2389222, upper bound: 372.2375286
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2389222, upper bound: 372.2375286
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2370422, upper bound: 372.2370422
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2370422, upper bound: 372.2375286
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2449358, upper bound: 372.2419277
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2399987
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2449358, upper bound: 372.2422569
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2403066, upper bound: 372.2403066
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2403066, upper bound: 372.2403066
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2375286, upper bound: 372.2389222
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.19
Output dim: 2, lower bound: -372.2375286, upper bound: 372.2403066

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -183.9405060, 146.2760010, -186.5588531, 148.3538971, -332.2944031, 332.8348389
1: -154.2132874, 129.5789032, -156.4776459, 131.4230652, -285.6363220, 286.0565491
2: -202.9273834, 131.7366943, -205.8432922, 133.5532837, -336.4806519, 337.5799866
3: -215.1352844, 113.7425995, -218.1713562, 115.3038177, -330.4390869, 331.9139404
4: -198.0909576, 151.3186493, -200.9426117, 153.4281616, -351.5191040, 352.2612610
5: -176.6286469, 137.1627808, -179.0869141, 139.0680237, -315.6966553, 316.2496643
6: -169.5669556, 163.1527557, -171.9492798, 165.5118866, -335.0788269, 335.1020203
7: -184.4985504, 154.7545166, -187.0674591, 156.9104309, -341.4089355, 341.8219299
8: -222.4814301, 152.7266693, -225.6847839, 154.8995361, -377.3808899, 378.4114380
9: -167.9209442, 165.3393860, -170.3481750, 167.7708130, -335.6917725, 335.6875610

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2403464, upper bound: 372.2392159
time: 7.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2403464, upper bound: 372.2392159
time: 7.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -181.2983093, 144.1922150, -187.0317993, 148.7418365, -330.0401001, 331.2239990
1: -151.9887390, 127.7242966, -156.7990112, 131.7157898, -283.7045288, 284.5233154
2: -200.0204163, 129.8491821, -206.3538971, 133.7982941, -333.8186951, 336.2030640
3: -212.0262451, 112.1058121, -218.6730652, 115.5043182, -327.5305786, 330.7788696
4: -195.2611694, 149.1485596, -201.4799042, 153.7298584, -348.9910278, 350.6284485
5: -174.0844116, 135.1913147, -179.4671783, 139.3339233, -313.4183350, 314.6584473
6: -167.1361237, 160.8145752, -172.3628693, 165.9335327, -333.0696411, 333.1773682
7: -181.8354034, 152.5259094, -187.4620514, 157.2367401, -339.0721436, 339.9879150
8: -219.3073730, 150.5562286, -226.2463837, 155.2071381, -374.5144958, 376.8026123
9: -165.5135193, 162.9730377, -170.7521820, 168.1990967, -333.7125549, 333.7251587

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2214677, upper bound: 372.2253816
time: 8.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2183247, upper bound: 372.2168366
time: 7.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -184.7630005, 146.9142303, -186.5588531, 148.3538971, -333.1168823, 333.4730530
1: -154.8224792, 130.1094666, -156.4776459, 131.4230652, -286.2455139, 286.5870972
2: -203.7982635, 132.2575531, -205.8432922, 133.5532837, -337.3515320, 338.1008301
3: -216.0290222, 114.1989517, -218.1713562, 115.3038177, -331.3328247, 332.3703003
4: -199.0048676, 151.9495850, -200.9426117, 153.4281616, -352.4329834, 352.8921814
5: -177.4187622, 137.7374268, -179.0869141, 139.0680237, -316.4867859, 316.8243408
6: -170.3013763, 163.8634491, -171.9492798, 165.5118866, -335.8132324, 335.8127441
7: -185.2854462, 155.4028168, -187.0674591, 156.9104309, -342.1958618, 342.4702454
8: -223.4143219, 153.3123474, -225.6847839, 154.8995361, -378.3138428, 378.9971313
9: -168.6418762, 166.0639954, -170.3481750, 167.7708130, -336.4126892, 336.4121399

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
time: 7.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
time: 7.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -182.1474609, 144.8519287, -187.0317993, 148.7418365, -330.8892822, 331.8837280
1: -152.6203918, 128.2741699, -156.7990112, 131.7157898, -284.3361816, 285.0731201
2: -200.9206696, 130.3895111, -206.3538971, 133.7982941, -334.7189636, 336.7433777
3: -212.9512024, 112.5797501, -218.6730652, 115.5043182, -328.4555054, 331.2528076
4: -196.2039490, 149.8021545, -201.4799042, 153.7298584, -349.9338074, 351.2820435
5: -174.9002533, 135.7863007, -179.4671783, 139.3339233, -314.2341614, 315.2534485
6: -167.8948822, 161.5487061, -172.3628693, 165.9335327, -333.8284302, 333.9115295
7: -182.6504517, 153.1971436, -187.4620514, 157.2367401, -339.8872070, 340.6591797
8: -220.2728271, 151.1638336, -226.2463837, 155.2071381, -375.4799805, 377.4102173
9: -166.2592468, 163.7225189, -170.7521820, 168.1990967, -334.4583435, 334.4747009

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2195961, upper bound: 372.2225976
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2151485, upper bound: 372.2123022
time: 8.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -185.3598480, 147.4630737, -187.6101227, 149.1786041, -334.5384521, 335.0731506
1: -155.4373627, 130.6425629, -157.2836456, 132.1199951, -287.5572510, 287.9262085
2: -204.4875946, 132.7601776, -206.9680786, 134.2395630, -338.7271729, 339.7282715
3: -216.8226776, 114.5881119, -219.3390350, 115.9040146, -332.7266541, 333.9270935
4: -199.5163422, 152.5262909, -202.1149902, 154.2523346, -353.7686768, 354.6412964
5: -177.9870911, 138.2009735, -180.0961304, 139.8169556, -317.8039856, 318.2971191
6: -170.9117279, 164.3899994, -172.9003296, 166.4309540, -337.3426819, 337.2903137
7: -185.8299255, 155.9126282, -188.0918274, 157.7535553, -343.5834961, 344.0044250
8: -224.2525787, 153.9574738, -226.9026337, 155.6739807, -379.9265747, 380.8600769
9: -169.1975555, 166.6198578, -171.2839203, 168.7038727, -337.9013977, 337.9037781

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2187822, upper bound: 372.2259678
time: 7.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2161239, upper bound: 372.2206312
time: 8.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -185.3598480, 147.4630737, -188.0523071, 149.5418854, -334.9016724, 335.5153198
1: -155.4373627, 130.6425629, -157.5830841, 132.3912354, -287.8285828, 288.2256165
2: -204.4875946, 132.7601776, -207.4473267, 134.4710388, -338.9586182, 340.2075195
3: -216.8226776, 114.5881119, -219.8019409, 116.0921021, -332.9147949, 334.3899841
4: -199.5163422, 152.5262909, -202.6236267, 154.5286713, -354.0449829, 355.1499023
5: -177.9870911, 138.2009735, -180.4466400, 140.0610199, -318.0480347, 318.6475830
6: -170.9117279, 164.3899994, -173.2885437, 166.8246002, -337.7362366, 337.6785278
7: -185.8299255, 155.9126282, -188.4638977, 158.0554504, -343.8853760, 344.3765259
8: -224.2525787, 153.9574738, -227.4293823, 155.9623108, -380.2149048, 381.3868408
9: -169.1975555, 166.6198578, -171.6560974, 169.1062164, -338.3037720, 338.2759399

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2187822, upper bound: 372.2259678
time: 7.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2161239, upper bound: 372.2206312
time: 7.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -186.1193085, 148.0758820, -182.3213806, 144.9939423, -331.1132507, 330.3972168
1: -155.9946136, 131.1351776, -152.7705078, 128.4023132, -284.3969116, 283.9057007
2: -205.3137207, 133.2117310, -201.1132660, 130.5157776, -335.8294983, 334.3250122
3: -217.6546326, 114.9604874, -213.1610870, 112.6900558, -330.3446960, 328.1215820
4: -200.3634796, 153.0643311, -196.4052277, 149.9486694, -350.3121033, 349.4695435
5: -178.6352539, 138.6867676, -175.0682831, 135.9193420, -314.5545349, 313.7550354
6: -171.5843506, 165.0631409, -168.0593262, 161.7095947, -333.2939453, 333.1224365
7: -186.5145416, 156.4786835, -182.8333435, 153.3479156, -339.8624573, 339.3120117
8: -225.1608582, 154.5166016, -220.4861298, 151.3054199, -376.4662781, 375.0026550
9: -169.8682098, 167.3036652, -166.4247284, 163.8807983, -333.7489929, 333.7283936

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 92

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2144709, upper bound: 372.2207854
time: 7.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2108724, upper bound: 372.2108724
time: 6.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -186.1193085, 148.0758820, -186.7843323, 148.5009918, -334.6203003, 334.8601685
1: -155.9946136, 131.1351776, -156.5524597, 131.5299835, -287.5245972, 287.6876221
2: -205.3137207, 133.2117310, -206.0325470, 133.6729431, -338.9866638, 339.2442627
3: -217.6546326, 114.9604874, -218.3728180, 115.4277420, -333.0823669, 333.3333130
4: -200.3634796, 153.0643311, -201.2225800, 153.5915527, -353.9549255, 354.2869263
5: -178.6352539, 138.6867676, -179.3173523, 139.2286987, -317.8639221, 318.0040894
6: -171.5843506, 165.0631409, -172.1662445, 165.6721954, -337.2565308, 337.2293701
7: -186.5145416, 156.4786835, -187.2982941, 157.0850067, -343.5995483, 343.7768860
8: -225.1608582, 154.5166016, -225.8836823, 154.9662628, -380.1271362, 380.4002686
9: -169.8682098, 167.3036652, -170.5198669, 167.9069824, -337.7751160, 337.8235474

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2144709, upper bound: 372.2214209
time: 8.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2108724, upper bound: 372.2115461
time: 7.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -188.1866608, 149.6107635, -186.7677460, 148.5179291, -336.7045898, 336.3785095
1: -157.8115234, 132.5553894, -156.6559143, 131.5695496, -289.3810730, 289.2112732
2: -207.6076660, 134.7412567, -206.0736694, 133.7010193, -341.3086853, 340.8149414
3: -220.0957794, 116.3474121, -218.4152832, 115.4319153, -335.5277100, 334.7626953
4: -202.6794434, 154.7826233, -201.1670380, 153.5979004, -356.2772827, 355.9496460
5: -180.6741180, 140.3087006, -179.2858429, 139.2230377, -319.8971558, 319.5944824
6: -173.4722290, 166.9274902, -172.1407623, 165.6979675, -339.1701355, 339.0682373
7: -188.7507629, 158.3126831, -187.2766418, 157.0855560, -345.8363037, 345.5892334
8: -227.6111908, 156.2059021, -225.9375000, 155.0708771, -382.6820068, 382.1433411
9: -171.8202515, 169.1698303, -170.5401611, 167.9602356, -339.7804871, 339.7099915

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2415328, upper bound: 372.2415328
time: 6.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2415328, upper bound: 372.2415328
time: 7.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -185.5578766, 147.5390015, -187.2426453, 148.9080963, -334.4659729, 334.7816467
1: -155.5986176, 130.7102509, -156.9790955, 131.8636475, -287.4622803, 287.6893005
2: -204.7155609, 132.8627777, -206.5864410, 133.9473572, -338.6629028, 339.4491272
3: -217.0029602, 114.7193832, -218.9196320, 115.6340408, -332.6369934, 333.6389771
4: -199.8649597, 152.6241455, -201.7064819, 153.9013977, -353.7663574, 354.3306274
5: -178.1437225, 138.3481293, -179.6684418, 139.4907074, -317.6343994, 318.0165100
6: -171.0534973, 164.6017914, -172.5563965, 166.1212006, -337.1746826, 337.1580811
7: -186.1011505, 156.0957031, -187.6731873, 157.4134827, -343.5146484, 343.7688599
8: -224.4539490, 154.0460663, -226.5017395, 155.3802948, -379.8341980, 380.5477600
9: -169.4257507, 166.8163452, -170.9460602, 168.3904114, -337.8161621, 337.7623901

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2229156, upper bound: 372.2279770
time: 7.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197851, upper bound: 372.2197851
time: 7.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -189.1215210, 150.3420410, -186.7677460, 148.5179291, -337.6394653, 337.1098022
1: -158.5181427, 133.1634979, -156.6559143, 131.5695496, -290.0876770, 289.8193970
2: -208.6014252, 135.3455353, -206.0736694, 133.7010193, -342.3024292, 341.4191895
3: -221.1181030, 116.8704834, -218.4152832, 115.4319153, -336.5500183, 335.2857666
4: -203.7108917, 155.5090027, -201.1670380, 153.5979004, -357.3087769, 356.6760254
5: -181.5682373, 140.9720917, -179.2858429, 139.2230377, -320.7912598, 320.2579346
6: -174.3113403, 167.7360535, -172.1407623, 165.6979675, -340.0092468, 339.8768311
7: -189.6472168, 159.0542145, -187.2766418, 157.0855560, -346.7327271, 346.3308411
8: -228.6891785, 156.8935089, -225.9375000, 155.0708771, -383.7600708, 382.8309937
9: -172.6422882, 170.0003510, -170.5401611, 167.9602356, -340.6024780, 340.5404968

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
time: 8.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
time: 8.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -186.5226898, 148.2926331, -187.2426453, 148.9080963, -335.4307861, 335.5352783
1: -156.3300781, 131.3394318, -156.9790955, 131.8636475, -288.1937256, 288.3184509
2: -205.7418823, 133.4882050, -206.5864410, 133.9473572, -339.6892090, 340.0745239
3: -218.0597534, 115.2607346, -218.9196320, 115.6340408, -333.6937866, 334.1803589
4: -200.9277954, 153.3738556, -201.7064819, 153.9013977, -354.8291931, 355.0803223
5: -179.0661621, 139.0332794, -179.6684418, 139.4907074, -318.5568237, 318.7017212
6: -171.9195557, 165.4361725, -172.5563965, 166.1212006, -338.0407715, 337.9925537
7: -187.0266418, 156.8622894, -187.6731873, 157.4134827, -344.4401245, 344.5354309
8: -225.5673523, 154.7566833, -226.5017395, 155.3802948, -380.9476318, 381.2583923
9: -170.2747345, 167.6732178, -170.9460602, 168.3904114, -338.6651306, 338.6192627

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2213728, upper bound: 372.2258586
time: 7.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2173335, upper bound: 372.2154105
time: 7.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -189.6658783, 150.8441162, -187.8294067, 149.3514099, -339.0172729, 338.6734619
1: -159.0864868, 133.6622772, -157.4708252, 132.2738342, -291.3602295, 291.1331177
2: -209.2348633, 135.8092041, -207.2099915, 134.3948975, -343.6297607, 343.0191956
3: -221.8514862, 117.2290649, -219.5952606, 116.0389404, -337.8903809, 336.8243408
4: -204.1699524, 156.0412292, -202.3502197, 154.4309998, -358.6009216, 358.3914490
5: -182.0859833, 141.3945770, -180.3050537, 139.9799194, -322.0659180, 321.6996155
6: -174.8741913, 168.2174225, -173.1016235, 166.6259613, -341.5001526, 341.3190308
7: -190.1438446, 159.5215149, -188.3114471, 157.9373779, -348.0812378, 347.8329163
8: -229.4601746, 157.4869385, -227.1685028, 155.8544312, -385.3146057, 384.6553955
9: -173.1528625, 170.5063782, -171.4853210, 168.9028015, -342.0556641, 341.9916687

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2203456, upper bound: 372.2288529
time: 7.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2178992, upper bound: 372.2239124
time: 7.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -189.6658783, 150.8441162, -188.2776642, 149.7193909, -339.3852539, 339.1216736
1: -159.0864868, 133.6622772, -157.7752533, 132.5493164, -291.6356506, 291.4375305
2: -209.2348633, 135.8092041, -207.6958466, 134.6302185, -343.8649902, 343.5050659
3: -221.8514862, 117.2290649, -220.0653839, 116.2305984, -338.0820618, 337.2944031
4: -204.1699524, 156.0412292, -202.8652344, 154.7120361, -358.8819275, 358.9064026
5: -182.0859833, 141.3945770, -180.6615143, 140.2283783, -322.3143311, 322.0560608
6: -174.8741913, 168.2174225, -173.4952240, 167.0250244, -341.8992310, 341.7126465
7: -190.1438446, 159.5215149, -188.6890259, 158.2442780, -348.3880920, 348.2105103
8: -229.4601746, 157.4869385, -227.7020721, 156.1471100, -385.6072693, 385.1890259
9: -173.1528625, 170.5063782, -171.8629456, 169.3104858, -342.4633484, 342.3693237

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2203456, upper bound: 372.2288529
time: 7.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2178992, upper bound: 372.2239124
time: 7.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -190.2883759, 151.3569641, -182.3213806, 144.9939423, -335.2823181, 333.6783447
1: -159.5312195, 134.0548706, -152.7705078, 128.4023132, -287.9335327, 286.8253784
2: -209.9085236, 136.1610870, -201.1132660, 130.5157776, -340.4243164, 337.2743225
3: -222.5278625, 117.5141068, -213.1610870, 112.6900558, -335.2179260, 330.6752014
4: -204.8663483, 156.4666290, -196.4052277, 149.9486694, -354.8150024, 352.8718567
5: -182.6097565, 141.7781830, -175.0682831, 135.9193420, -318.5290833, 316.8464661
6: -175.4176483, 168.7714081, -168.0593262, 161.7095947, -337.1272278, 336.8307495
7: -190.6843567, 159.9704742, -182.8333435, 153.3479156, -344.0322571, 342.8038330
8: -230.2047882, 157.9357147, -220.4861298, 151.3054199, -381.5101929, 378.4218140
9: -173.6960907, 171.0709381, -166.4247284, 163.8807983, -337.5769043, 337.4956665

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 92

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2150815, upper bound: 372.2225488
time: 8.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2115461, upper bound: 372.2128853
time: 7.29 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -190.2883759, 151.3569641, -186.7843323, 148.5009918, -338.7893677, 338.1412964
1: -159.5312195, 134.0548706, -156.5524597, 131.5299835, -291.0612183, 290.6073303
2: -209.9085236, 136.1610870, -206.0325470, 133.6729431, -343.5814514, 342.1936340
3: -222.5278625, 117.5141068, -218.3728180, 115.4277420, -337.9555969, 335.8869324
4: -204.8663483, 156.4666290, -201.2225800, 153.5915527, -358.4578552, 357.6892090
5: -182.6097565, 141.7781830, -179.3173523, 139.2286987, -321.8384399, 321.0954895
6: -175.4176483, 168.7714081, -172.1662445, 165.6721954, -341.0898438, 340.9376526
7: -190.6843567, 159.9704742, -187.2982941, 157.0850067, -347.7693481, 347.2687683
8: -230.2047882, 157.9357147, -225.8836823, 154.9662628, -385.1710510, 383.8193970
9: -173.6960907, 171.0709381, -170.5198669, 167.9069824, -341.6030579, 341.5908203

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2150815, upper bound: 372.2249188
time: 6.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2115461, upper bound: 372.2146912
time: 7.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.07 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2403464, upper bound: 372.2392159
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2403464, upper bound: 372.2392159
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2214677, upper bound: 372.2253816
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2183247, upper bound: 372.2168366
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2195961, upper bound: 372.2225976
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2151485, upper bound: 372.2123022
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2187822, upper bound: 372.2259678
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2161239, upper bound: 372.2206312
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2187822, upper bound: 372.2259678
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2161239, upper bound: 372.2206312
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2144709, upper bound: 372.2207854
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2108724, upper bound: 372.2108724
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2144709, upper bound: 372.2214209
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2108724, upper bound: 372.2115461
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2415328, upper bound: 372.2415328
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2415328, upper bound: 372.2415328
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2229156, upper bound: 372.2279770
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2197851, upper bound: 372.2197851
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2213728, upper bound: 372.2258586
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2173335, upper bound: 372.2154105
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2203456, upper bound: 372.2288529
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2178992, upper bound: 372.2239124
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2203456, upper bound: 372.2288529
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2178992, upper bound: 372.2239124
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2150815, upper bound: 372.2225488
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2115461, upper bound: 372.2128853
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2150815, upper bound: 372.2249188
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.07
Output dim: 2, lower bound: -372.2115461, upper bound: 372.2146912

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -176.2814026, 140.2556915, -186.5588531, 148.3538971, -324.6353149, 326.8145142
1: -147.7618256, 124.2178192, -156.4776459, 131.4230652, -279.1848755, 280.6954651
2: -194.5114288, 126.2875366, -205.8432922, 133.5532837, -328.0646973, 332.1308289
3: -206.1377106, 109.0051651, -218.1713562, 115.3038177, -321.4415283, 327.1765137
4: -189.9000854, 145.0474701, -200.9426117, 153.4281616, -343.3281860, 345.9900818
5: -169.2700195, 131.4541626, -179.0869141, 139.0680237, -308.3380432, 310.5410767
6: -162.5184174, 156.3793182, -171.9492798, 165.5118866, -328.0303040, 328.3285828
7: -176.7870789, 148.2987213, -187.0674591, 156.9104309, -333.6974792, 335.3661499
8: -213.2905579, 146.4603119, -225.6847839, 154.8995361, -368.1900940, 372.1450806
9: -160.9387665, 158.4847717, -170.3481750, 167.7708130, -328.7095947, 328.8329468

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2406378
time: 8.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2409720
time: 8.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -176.7277832, 140.6148682, -186.5588531, 148.3538971, -325.0816650, 327.1736755
1: -148.0576324, 124.4867172, -156.4776459, 131.4230652, -279.4806824, 280.9643555
2: -194.9902649, 126.5123596, -205.8432922, 133.5532837, -328.5434570, 332.3556519
3: -206.6006012, 109.1849213, -218.1713562, 115.3038177, -321.9043884, 327.3562622
4: -190.3997345, 145.3235321, -200.9426117, 153.4281616, -343.8278809, 346.2661438
5: -169.6182251, 131.6971436, -179.0869141, 139.0680237, -308.6862488, 310.7840271
6: -162.9032288, 156.7723846, -171.9492798, 165.5118866, -328.4151001, 328.7216492
7: -177.1501923, 148.5994568, -187.0674591, 156.9104309, -334.0605774, 335.6668701
8: -213.8143311, 146.7455597, -225.6847839, 154.8995361, -368.7138672, 372.4303589
9: -161.3145599, 158.8863983, -170.3481750, 167.7708130, -329.0853882, 329.2345276

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2406378
time: 8.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2409720
time: 8.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -172.9277802, 137.5520782, -187.0317993, 148.7418365, -321.6695862, 324.5838623
1: -144.8802643, 121.8014984, -156.7990112, 131.7157898, -276.5960693, 278.6004639
2: -190.7637787, 123.7447815, -206.3538971, 133.7982941, -324.5620422, 330.0986938
3: -202.1618652, 106.8938751, -218.6730652, 115.5043182, -317.6661682, 325.5668945
4: -186.2725220, 142.1981812, -201.4799042, 153.7298584, -340.0023804, 343.6780701
5: -166.0064392, 128.9039459, -179.4671783, 139.3339233, -305.3403625, 308.3711243
6: -159.3984375, 153.3902435, -172.3628693, 165.9335327, -325.3319397, 325.7530212
7: -173.3770447, 145.4338989, -187.4620514, 157.2367401, -330.6137695, 332.8959045
8: -209.2037964, 143.5526886, -226.2463837, 155.2071381, -364.4109497, 369.7990417
9: -157.8524323, 155.4603577, -170.7521820, 168.1990967, -326.0514832, 326.2124634

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 205

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197229, upper bound: 372.2250114
time: 7.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197229, upper bound: 372.2253816
time: 7.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -177.0718842, 140.8677216, -186.5588531, 148.3538971, -325.4257812, 327.4265442
1: -148.3450623, 124.7241898, -156.4776459, 131.4230652, -279.7680969, 281.2018433
2: -195.3483887, 126.7834549, -205.8432922, 133.5532837, -328.9016418, 332.6267395
3: -206.9955597, 109.4398193, -218.1713562, 115.3038177, -322.2993774, 327.6111145
4: -190.7790833, 145.6529083, -200.9426117, 153.4281616, -344.2072449, 346.5955200
5: -170.0284271, 132.0045624, -179.0869141, 139.0680237, -309.0964355, 311.0914917
6: -163.2245331, 157.0619812, -171.9492798, 165.5118866, -328.7363892, 329.0112610
7: -177.5425262, 148.9178009, -187.0674591, 156.9104309, -334.4529419, 335.9852600
8: -214.1867065, 147.0196838, -225.6847839, 154.8995361, -369.0862427, 372.7044678
9: -161.6298981, 159.1813965, -170.3481750, 167.7708130, -329.4006958, 329.5295715

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2421257, upper bound: 372.2390469
time: 8.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2421257, upper bound: 372.2393725
time: 7.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 17.53 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2406378
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2409720
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2406378
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2431695, upper bound: 372.2409720
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2197229, upper bound: 372.2250114
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2197229, upper bound: 372.2253816
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2421257, upper bound: 372.2390469
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 17.53
Output dim: 2, lower bound: -372.2421257, upper bound: 372.2393725
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2391156, upper bound: 372.2374964
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2195961, upper bound: 372.2225976
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2187822, upper bound: 372.2259678
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2161239, upper bound: 372.2206312
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2187822, upper bound: 372.2259678
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2161239, upper bound: 372.2206312
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2144709, upper bound: 372.2207854
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2144709, upper bound: 372.2214209
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2415328, upper bound: 372.2415328
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2415328, upper bound: 372.2415328
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2229156, upper bound: 372.2279770
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2197851, upper bound: 372.2197851
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2405816, upper bound: 372.2402591
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2213728, upper bound: 372.2258586
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2203456, upper bound: 372.2288529
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2178992, upper bound: 372.2239124
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2203456, upper bound: 372.2288529
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2178992, upper bound: 372.2239124
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2150815, upper bound: 372.2225488
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 2, lower bound: -372.2150815, upper bound: 372.2249188
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=375.07427978515625
rel_dist={2: [-372.26984149889176, 372.2698414979088]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2678443, upper bound: 372.2672729
time: 9.65 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2687117, upper bound: 372.2687117
time: 9.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.34
Output dim: 2, lower bound: -372.2678443, upper bound: 372.2672729
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.34
Output dim: 2, lower bound: -372.2687117, upper bound: 372.2687117

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -195.3206635, 155.3128815, -202.5672760, 161.0227966, -356.3434448, 357.8800659
1: -163.8407288, 137.6345367, -169.9779205, 142.7146912, -306.5554199, 307.6124573
2: -215.4388123, 139.8561096, -223.4282379, 144.9801483, -360.4188843, 363.2843018
3: -228.5296783, 120.7476654, -237.0142975, 125.1877747, -353.7174683, 357.7619629
4: -210.1535492, 160.6972504, -217.9472198, 166.6109467, -376.7644958, 378.6444092
5: -187.5573120, 145.6229858, -194.4796295, 150.9902954, -338.5475769, 340.1026001
6: -180.0850830, 173.1997681, -186.7381287, 179.6372528, -359.7223511, 359.9378967
7: -195.8436737, 164.3047943, -203.0926514, 170.3765869, -366.2201843, 367.3973694
8: -236.2275085, 162.1352081, -244.9648285, 168.0840149, -404.3115234, 407.1000366
9: -178.2790680, 175.5395508, -184.9122772, 182.0818329, -360.3608398, 360.4518433

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2568078, upper bound: 372.2554201
time: 8.91 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2566912, upper bound: 372.2558761
time: 11.03 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -199.6509399, 158.7141571, -203.6721039, 161.8899384, -361.5408936, 362.3862000
1: -167.5093231, 140.6702576, -170.9206390, 143.4893188, -310.9985962, 311.5908508
2: -220.2118988, 142.9197845, -224.6456451, 145.7613373, -365.9732361, 367.5653687
3: -233.5867157, 123.4031067, -238.3024139, 125.8665161, -359.4532166, 361.7055054
4: -214.8302155, 164.2307281, -219.1323547, 167.5086212, -382.3388367, 383.3630981
5: -191.6817169, 148.8310699, -195.5306702, 151.8097229, -343.4914551, 344.3617554
6: -184.0677643, 177.0479431, -187.7510071, 180.6207581, -364.6885376, 364.7989502
7: -200.1786194, 167.9325256, -204.1981201, 171.3022614, -371.4808960, 372.1306458
8: -241.4618378, 165.6829529, -246.3032837, 168.9886322, -410.4504700, 411.9862366
9: -182.2528229, 179.4474182, -185.9271851, 183.0834961, -365.3363037, 365.3746033

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2586983, upper bound: 372.2582822
time: 11.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2586266, upper bound: 372.2586266
time: 7.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.00
Output dim: 2, lower bound: -372.2568078, upper bound: 372.2554201
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.00
Output dim: 2, lower bound: -372.2566912, upper bound: 372.2558761
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.00
Output dim: 2, lower bound: -372.2586983, upper bound: 372.2582822
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.00
Output dim: 2, lower bound: -372.2586266, upper bound: 372.2586266

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -192.0910797, 152.7482452, -191.1592102, 151.9637604, -344.0548401, 343.9074707
1: -161.1090393, 135.3482819, -160.3273468, 134.6394043, -295.7484436, 295.6756287
2: -211.8878937, 137.5517731, -210.8862610, 136.8412628, -348.7290344, 348.4380493
3: -224.7280731, 118.7602692, -223.5874176, 118.1660385, -342.8941040, 342.3476868
4: -206.7293396, 158.0355988, -205.8552856, 157.2091980, -363.9385071, 363.8908691
5: -184.4558258, 143.2222137, -183.5238800, 142.5091248, -326.9649048, 326.7460938
6: -177.0999756, 170.3483124, -176.1943207, 169.5657196, -346.6656189, 346.5426025
7: -192.6235809, 161.5948181, -191.7198944, 160.8028259, -353.4263916, 353.3146667
8: -232.3258514, 159.4645691, -231.1838684, 158.6530304, -390.9788818, 390.6484375
9: -175.3392029, 172.6448212, -174.5291138, 171.8561554, -347.1953735, 347.1738892

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2395721, upper bound: 372.2397650
time: 9.43 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2381155, upper bound: 372.2369273
time: 8.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -188.7470703, 150.0941010, -192.0234985, 152.6349640, -341.3820190, 342.1176147
1: -158.2636566, 132.9790344, -160.9709473, 135.1999512, -293.4635315, 293.9499817
2: -208.1996765, 135.1572571, -211.8023987, 137.3926849, -345.5923462, 346.9596558
3: -220.7911530, 116.6862564, -224.5281219, 118.6486511, -339.4397583, 341.2143250
4: -203.1710510, 155.2715759, -206.8132477, 157.8749695, -361.0460205, 362.0848389
5: -181.2471161, 140.7304840, -184.3529358, 143.1160889, -324.3632202, 325.0834351
6: -174.0118408, 167.3880768, -176.9671631, 170.3119965, -344.3238525, 344.3551636
7: -189.2744141, 158.7801971, -192.5478516, 161.4865112, -350.7608337, 351.3280029
8: -228.2711792, 156.6786041, -232.1685944, 159.2742157, -387.5454102, 388.8471985
9: -172.2858124, 169.6428986, -175.2877045, 172.6175995, -344.9034119, 344.9306030

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2396969, upper bound: 372.2401018
time: 8.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2382769, upper bound: 372.2372375
time: 8.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -196.4006500, 156.1329193, -192.2248230, 152.7999725, -349.2006226, 348.3577271
1: -164.7599792, 138.3693695, -161.2373657, 135.3867798, -300.1467590, 299.6067200
2: -216.6377563, 140.6011810, -212.0613708, 137.5945892, -354.2323608, 352.6625366
3: -229.7618408, 121.4033356, -224.8310089, 118.8209305, -348.5827637, 346.2343445
4: -211.3848267, 161.5519714, -206.9997406, 158.0743103, -369.4590759, 368.5516968
5: -188.5606537, 146.4149475, -184.5379486, 143.3000793, -331.8607178, 330.9528503
6: -181.0632782, 174.1783142, -177.1710663, 170.5152283, -351.5784912, 351.3493652
7: -196.9386597, 165.2052460, -192.7870483, 161.6961212, -358.6347656, 357.9923096
8: -237.5344543, 162.9951935, -232.4738922, 159.5260162, -397.0604553, 395.4689941
9: -179.2947845, 176.5336914, -175.5096436, 172.8218536, -352.1166382, 352.0433350

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2412956, upper bound: 372.2427063
time: 7.90 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2401564, upper bound: 372.2398987
time: 7.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -193.1228943, 153.5328369, -193.1667480, 153.5343933, -346.6572876, 346.6995850
1: -161.9718781, 136.0465240, -161.9477386, 136.0013885, -297.9732361, 297.9942627
2: -213.0245667, 138.2542572, -213.0634155, 138.2015076, -351.2260132, 351.3175964
3: -225.9012299, 119.3702545, -225.8629913, 119.3517303, -345.2529602, 345.2332153
4: -207.8978882, 158.8441620, -208.0397186, 158.8057251, -366.7036133, 366.8838806
5: -185.4150085, 143.9735565, -185.4403076, 143.9655609, -329.3805542, 329.4138794
6: -178.0390778, 171.2765350, -178.0176849, 171.3287354, -349.3677979, 349.2942200
7: -193.6553192, 162.4458771, -193.6919861, 162.4438934, -356.0992126, 356.1378784
8: -233.5641479, 160.2674866, -233.5569916, 160.2145538, -393.7786865, 393.8244629
9: -176.3021545, 173.5924377, -176.3378296, 173.6552734, -349.9574280, 349.9302368

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2413115, upper bound: 372.2430683
time: 8.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2402199, upper bound: 372.2402199
time: 7.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2395721, upper bound: 372.2397650
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2381155, upper bound: 372.2369273
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2396969, upper bound: 372.2401018
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2382769, upper bound: 372.2372375
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2412956, upper bound: 372.2427063
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2401564, upper bound: 372.2398987
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2413115, upper bound: 372.2430683
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.53
Output dim: 2, lower bound: -372.2402199, upper bound: 372.2402199

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3807373, 146.6863251, -190.5859528, 151.5130310, -335.8937683, 337.2722778
1: -154.6145935, 129.9506989, -159.8446045, 134.2381744, -288.8527832, 289.7952881
2: -203.4151917, 132.0648346, -210.2563171, 136.4334412, -339.8486328, 342.3211365
3: -215.6705627, 113.9915619, -222.9138184, 117.8115387, -333.4820557, 336.9053955
4: -198.4838715, 151.7222900, -205.2423706, 156.7397766, -355.2236328, 356.9646301
5: -177.0464630, 137.4755859, -182.9729004, 142.0818634, -319.1283264, 320.4484863
6: -170.0051727, 163.5287323, -175.6668243, 169.0587158, -339.0639038, 339.1955261
7: -184.8596649, 155.0947876, -191.1427307, 160.3197021, -345.1793213, 346.2375183
8: -223.0744934, 153.1551666, -230.4961853, 158.1839142, -381.2583618, 383.6513672
9: -168.3101044, 165.7445068, -174.0066223, 171.3431549, -339.6531982, 339.7511292

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2381155, upper bound: 372.2369273
time: 9.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2381155, upper bound: 372.2369273
time: 8.88 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -185.1128082, 147.2763672, -184.6159210, 146.8047180, -331.9175110, 331.8922729
1: -155.1485748, 130.4238586, -154.8176270, 130.0471344, -285.1956177, 285.2414551
2: -204.2112885, 132.4946899, -203.6870728, 132.1656799, -336.3769226, 336.1817627
3: -216.4708405, 114.3459167, -215.8889618, 114.1126404, -330.5834961, 330.2348633
4: -199.3002014, 152.2366180, -198.8477783, 151.8351746, -351.1353760, 351.0844116
5: -177.6679230, 137.9393463, -177.2250061, 137.6268311, -315.2946777, 315.1643372
6: -170.6522675, 164.1778107, -170.1733398, 163.7754211, -334.4276733, 334.3511353
7: -185.5143585, 155.6366577, -185.1249695, 155.2844696, -340.7988281, 340.7615967
8: -223.9493561, 153.6893921, -223.3220520, 153.2763672, -377.2257080, 377.0114441
9: -168.9541321, 166.4027557, -168.5668030, 165.9962311, -334.9503784, 334.9695435

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2182331, upper bound: 372.2132725
time: 9.10 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2133090, upper bound: 372.2115015
time: 8.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -181.0280609, 144.0252228, -191.4491730, 152.1832733, -333.2113342, 335.4743042
1: -151.7621155, 127.5749435, -160.4872894, 134.7978668, -286.5599976, 288.0621948
2: -199.7181549, 129.6637726, -211.1714172, 136.9838715, -336.7020264, 340.8352051
3: -211.7233124, 111.9108429, -223.8534393, 118.2933350, -330.0166321, 335.7641602
4: -194.9156799, 148.9514923, -206.1991577, 157.4047394, -352.3204041, 355.1506042
5: -173.8299561, 134.9774170, -183.8010406, 142.6879883, -316.5178833, 318.7784424
6: -166.9096832, 160.5612640, -176.4387665, 169.8041534, -336.7138062, 337.0000305
7: -181.5023956, 152.2719727, -191.9695587, 161.0022430, -342.5045776, 344.2415161
8: -219.0101013, 150.3620758, -231.4796906, 158.8043365, -377.8143616, 381.8417664
9: -165.2487030, 162.7343445, -174.7642670, 172.1036377, -337.3522644, 337.4985657

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2382769, upper bound: 372.2372375
time: 8.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2382769, upper bound: 372.2372375
time: 8.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -181.6097260, 144.4908752, -185.5422363, 147.5254669, -329.1351624, 330.0331116
1: -152.1717834, 127.9393234, -155.5150146, 130.6521301, -282.8239136, 283.4543457
2: -200.3440704, 129.9915161, -204.6728516, 132.7638245, -333.1079102, 334.6643677
3: -212.3445740, 112.1756592, -216.9034424, 114.6359406, -326.9805298, 329.0791016
4: -195.5648193, 149.3413239, -199.8739471, 152.5521088, -348.1169434, 349.2152710
5: -174.3087616, 135.3246918, -178.1131897, 138.2810822, -312.5898438, 313.4378662
6: -167.4189148, 161.0731049, -171.0045624, 164.5772858, -331.9961243, 332.0776672
7: -182.0068359, 152.6895142, -186.0179749, 156.0217285, -338.0285645, 338.7074890
8: -219.6984406, 150.7654572, -224.3850708, 153.9498749, -373.6483154, 375.1504517
9: -165.7519226, 163.2581787, -169.3836517, 166.8164215, -332.5683594, 332.6417542

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2178543, upper bound: 372.2130696
time: 11.47 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2120998, upper bound: 372.2111239
time: 7.99 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -188.6492767, 150.0362549, -191.6513519, 152.3488770, -340.9981689, 341.6875305
1: -158.2316742, 132.9434509, -160.7543335, 134.9854431, -293.2171021, 293.6977844
2: -208.1198425, 135.0873871, -211.4311981, 137.1866150, -345.3064575, 346.5185852
3: -220.6560822, 116.6087875, -224.1569977, 118.4662781, -339.1223450, 340.7657776
4: -203.0974274, 155.2052002, -206.3865509, 157.6047211, -360.7021179, 361.5917053
5: -181.1091614, 140.6407013, -183.9866028, 142.8726501, -323.9817810, 324.6272583
6: -173.9324646, 167.3225861, -176.6433716, 170.0080719, -343.9404602, 343.9659119
7: -189.1358490, 158.6724701, -192.2097015, 161.2126923, -350.3485413, 350.8821411
8: -228.2349701, 156.6519470, -231.7858429, 159.0568695, -387.2918396, 388.4377441
9: -172.2313080, 169.5962067, -174.9869843, 172.3085327, -344.5398560, 344.5830994

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2401564, upper bound: 372.2398987
time: 7.59 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2401564, upper bound: 372.2398987
time: 7.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -189.2396545, 150.5247192, -185.6857452, 147.6464233, -336.8860779, 336.2104492
1: -158.6497192, 133.3136444, -155.7322388, 130.7974701, -289.4471130, 289.0458069
2: -208.7594757, 135.4150543, -204.8671112, 132.9221649, -341.6816406, 340.2821045
3: -221.2950287, 116.8741074, -217.1379242, 114.7709427, -336.0659790, 334.0120239
4: -203.7585297, 155.6047363, -199.9977417, 152.7040558, -356.4625854, 355.6024780
5: -181.6027527, 141.0006409, -178.2446289, 138.4221649, -320.0248718, 319.2452698
6: -174.4472504, 167.8485718, -171.1548462, 164.7289581, -339.1762085, 339.0033569
7: -189.6427917, 159.0939636, -186.1964569, 156.1821747, -345.8249512, 345.2904053
8: -228.9432068, 157.0741730, -224.6185455, 154.1528320, -383.0960388, 381.6926575
9: -172.7447205, 170.1324310, -169.5522308, 166.9671936, -339.7119141, 339.6846619

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2214114, upper bound: 372.2167047
time: 8.45 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2160821, upper bound: 372.2149706
time: 8.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -185.3669128, 147.4334106, -192.5914459, 153.0819244, -338.4488525, 340.0248413
1: -155.4396210, 130.6176758, -161.4632568, 135.5986786, -291.0382996, 292.0808716
2: -204.5019836, 132.7369843, -212.4312897, 137.7920074, -342.2939758, 345.1682434
3: -216.7909698, 114.5724945, -225.1870117, 118.9959030, -335.7868652, 339.7594910
4: -199.6049042, 152.4937897, -207.4245911, 158.3346252, -357.9395142, 359.9183350
5: -177.9606171, 138.1950226, -184.8873291, 143.5366821, -321.4973145, 323.0823364
6: -170.9030914, 164.4172516, -177.4883881, 170.8199463, -341.7229309, 341.9056396
7: -185.8484650, 155.9080811, -193.1128082, 161.9588165, -347.8072815, 349.0208130
8: -224.2590790, 153.9212036, -232.8667145, 159.7438354, -384.0029297, 386.7879028
9: -169.2338867, 166.6506805, -175.8134613, 173.1404114, -342.3742981, 342.4641418

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2402199, upper bound: 372.2402199
time: 9.74 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2402199, upper bound: 372.2402199
time: 7.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -185.8217468, 147.8057251, -186.7017059, 148.4375000, -334.2591858, 334.5074158
1: -155.7444000, 130.8892822, -156.5057220, 131.4642029, -287.2086182, 287.3949585
2: -204.9860840, 132.9710693, -205.9513550, 133.5829010, -338.5689697, 338.9224243
3: -217.2688141, 114.7554092, -218.2573395, 115.3484039, -332.6172180, 333.0127258
4: -200.1125946, 152.7782288, -201.1178436, 153.4956055, -353.6082153, 353.8960571
5: -178.3233795, 138.4475861, -179.2166290, 139.1427917, -317.4661865, 317.6642151
6: -171.2908783, 164.8189850, -172.0693665, 165.6082001, -336.8990784, 336.8883362
7: -186.2177582, 156.2166138, -187.1768188, 156.9924011, -343.2101440, 343.3934326
8: -224.7941437, 154.2200470, -225.7921906, 154.9017029, -379.6958618, 380.0122375
9: -169.6181946, 167.0639038, -170.4481659, 167.8684235, -337.4865723, 337.5120850

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2211773, upper bound: 372.2166357
time: 11.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2146453, upper bound: 372.2146453
time: 7.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.12 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2381155, upper bound: 372.2369273
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2381155, upper bound: 372.2369273
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2182331, upper bound: 372.2132725
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2133090, upper bound: 372.2115015
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2382769, upper bound: 372.2372375
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2382769, upper bound: 372.2372375
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2178543, upper bound: 372.2130696
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2120998, upper bound: 372.2111239
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2401564, upper bound: 372.2398987
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2401564, upper bound: 372.2398987
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2214114, upper bound: 372.2167047
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2160821, upper bound: 372.2149706
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2402199, upper bound: 372.2402199
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2402199, upper bound: 372.2402199
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2211773, upper bound: 372.2166357
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 20.12
Output dim: 2, lower bound: -372.2146453, upper bound: 372.2146453

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.3807373, 146.6863251, -183.4777374, 145.9250031, -330.3057251, 330.1640320
1: -154.6145935, 129.9506989, -153.8575439, 129.2626953, -283.8772888, 283.8082275
2: -203.4151917, 132.0648346, -202.4454956, 131.3758240, -334.7910156, 334.5103149
3: -215.6705627, 113.9915619, -214.5640106, 113.4147110, -329.0852661, 328.5555420
4: -198.4838715, 151.7222900, -197.6416626, 150.9197388, -349.4035950, 349.3638916
5: -177.0464630, 137.4755859, -176.1431732, 136.7849274, -313.8313904, 313.6187744
6: -170.0051727, 163.5287323, -169.1258392, 162.7723389, -332.7774658, 332.6545715
7: -184.8596649, 155.0947876, -183.9860992, 154.3285828, -339.1882324, 339.0808716
8: -223.0744934, 153.1551666, -221.9673462, 152.3686676, -375.4430847, 375.1224976
9: -168.3101044, 165.7445068, -167.5275269, 164.9814453, -333.2915039, 333.2720337

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2173816, upper bound: 372.2203715
time: 10.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2154256, upper bound: 372.2170443
time: 7.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -184.3807373, 146.6863251, -183.8363647, 146.2177124, -330.5984192, 330.5227051
1: -154.6145935, 129.9506989, -154.0798340, 129.4710541, -284.0856323, 284.0305176
2: -203.4151917, 132.0648346, -202.8291321, 131.5380096, -334.9531860, 334.8939819
3: -215.6705627, 113.9915619, -214.9268494, 113.5421448, -329.2126770, 328.9183960
4: -198.4838715, 151.7222900, -198.0462952, 151.1246338, -349.6084290, 349.7685852
5: -177.0464630, 137.4755859, -176.4114380, 136.9625702, -314.0090332, 313.8869934
6: -170.0051727, 163.5287323, -169.4309998, 163.0884399, -333.0936279, 332.9597168
7: -184.8596649, 155.0947876, -184.2607117, 154.5556946, -339.4153442, 339.3554993
8: -223.0744934, 153.1551666, -222.3878326, 152.5826111, -375.6570435, 375.5429993
9: -168.3101044, 165.7445068, -167.8225403, 165.3051758, -333.6152954, 333.5670166

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2173816, upper bound: 372.2203715
time: 10.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2154256, upper bound: 372.2170443
time: 9.30 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -181.0280609, 144.0252228, -184.3199768, 146.5782318, -327.6062622, 328.3451233
1: -151.7621155, 127.5749435, -154.4832916, 129.8061218, -281.5682373, 282.0582275
2: -199.7181549, 129.6637726, -203.3389893, 131.9097900, -331.6279297, 333.0027466
3: -211.7233124, 111.9108429, -215.4799957, 113.8822861, -325.6055603, 327.3907776
4: -194.9156799, 148.9514923, -198.5750427, 151.5678864, -346.4835510, 347.5264587
5: -173.8299561, 134.9774170, -176.9505005, 137.3737488, -311.2036133, 311.9278564
6: -166.9096832, 160.5612640, -169.8791962, 163.4997864, -330.4094238, 330.4404297
7: -181.5023956, 152.2719727, -184.7921600, 154.9911499, -336.4934998, 337.0641479
8: -219.0101013, 150.3620758, -222.9277039, 152.9713898, -371.9814148, 373.2897949
9: -165.2487030, 162.7343445, -168.2653656, 165.7242126, -330.9728699, 330.9996033

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2154357, upper bound: 372.2198634
time: 9.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2139083, upper bound: 372.2166851
time: 9.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -181.0280609, 144.0252228, -184.6384735, 146.8402710, -327.8683167, 328.6635742
1: -151.7621155, 127.5749435, -154.6761475, 129.9862213, -281.7483521, 282.2510986
2: -199.7181549, 129.6637726, -203.6807098, 132.0524750, -331.7706299, 333.3444824
3: -211.7233124, 111.9108429, -215.7930298, 113.9918594, -325.7151489, 327.7037964
4: -194.9156799, 148.9514923, -198.9409790, 151.7402954, -346.6559448, 347.8923950
5: -173.8299561, 134.9774170, -177.1809387, 137.5233917, -311.3533020, 312.1583252
6: -166.9096832, 160.5612640, -170.1506805, 163.7789459, -330.6885681, 330.7119446
7: -181.5023956, 152.2719727, -185.0357971, 155.1863403, -336.6886292, 337.3077698
8: -219.0101013, 150.3620758, -223.3028870, 153.1597900, -372.1697998, 373.6649780
9: -165.2487030, 162.7343445, -168.5202637, 166.0140839, -331.2626648, 331.2545776

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2154357, upper bound: 372.2198634
time: 9.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2139083, upper bound: 372.2166851
time: 10.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -188.6492767, 150.0362549, -184.5321045, 146.7514801, -335.4007568, 334.5682678
1: -158.2316742, 132.9434509, -154.7585907, 130.0022888, -288.2339478, 287.7020264
2: -208.1198425, 135.0873871, -203.6081848, 132.1215363, -340.2413940, 338.6955261
3: -220.6560822, 116.6087875, -215.7936859, 114.0623093, -334.7182922, 332.4024353
4: -203.0974274, 155.2052002, -198.7746277, 151.7758636, -354.8732910, 353.9797668
5: -181.1091614, 140.6407013, -177.1454315, 137.5675354, -318.6766663, 317.7861328
6: -173.9324646, 167.3225861, -170.0928955, 163.7116547, -337.6441040, 337.4154358
7: -189.1358490, 158.6724701, -185.0421906, 155.2124023, -344.3481750, 343.7145996
8: -228.2349701, 156.6519470, -223.2445526, 153.2329865, -381.4679565, 379.8964539
9: -172.2313080, 169.5962067, -168.4979858, 165.9375916, -338.1688232, 338.0941772

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197532, upper bound: 372.2241287
time: 9.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2180359, upper bound: 372.2203429
time: 8.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -188.6492767, 150.0362549, -184.9104614, 147.0640259, -335.7133179, 334.9466248
1: -158.2316742, 132.9434509, -154.9986877, 130.2237854, -288.4554443, 287.9421387
2: -208.1198425, 135.0873871, -204.0140533, 132.2968445, -340.4166870, 339.1014404
3: -220.6560822, 116.6087875, -216.1823273, 114.2035294, -334.8595886, 332.7911072
4: -203.0974274, 155.2052002, -199.2008820, 151.9983521, -355.0957642, 354.4059753
5: -181.1091614, 140.6407013, -177.4355469, 137.7615509, -318.8706665, 318.0762329
6: -173.9324646, 167.3225861, -170.4168854, 164.0451508, -337.9776001, 337.7394104
7: -189.1358490, 158.6724701, -185.3367462, 155.4561615, -344.5920105, 344.0091248
8: -228.2349701, 156.6519470, -223.6902924, 153.4637909, -381.6987610, 380.3421936
9: -172.2313080, 169.5962067, -168.8114471, 166.2811584, -338.5123901, 338.4076233

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2197532, upper bound: 372.2241287
time: 10.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2180359, upper bound: 372.2203429
time: 8.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -187.7031555, 149.3056030, -177.3296509, 141.0177612, -328.7208862, 326.6352234
1: -157.3442993, 132.2260437, -148.6349792, 124.8843307, -282.2285767, 280.8610229
2: -207.0596313, 134.2945404, -195.6253967, 126.8293991, -333.8889160, 329.9198608
3: -219.4840240, 115.9164124, -207.2911530, 109.5661545, -329.0501709, 323.2075806
4: -202.1085205, 154.3285828, -191.0250854, 145.7673645, -347.8758850, 345.3536682
5: -180.1199951, 139.8461914, -170.1816406, 132.1453094, -312.2653198, 310.0278320
6: -173.0272064, 166.4850159, -163.4322815, 157.3162079, -330.3434143, 329.9172363
7: -188.0894165, 157.7914429, -177.7522888, 149.1012115, -337.1905518, 335.5437012
8: -227.0882111, 155.7884979, -214.5334930, 147.1644135, -374.2526245, 370.3219910
9: -171.3379211, 168.7530518, -161.9041748, 159.4665222, -330.8043823, 330.6572266

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 133

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1993581, upper bound: 372.1974775
time: 9.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1988048, upper bound: 372.1941774
time: 8.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -185.3669128, 147.4334106, -185.4497833, 147.4670715, -332.8339539, 332.8831787
1: -155.4396210, 130.6176758, -155.4491577, 130.5988770, -286.0385132, 286.0667725
2: -204.5019836, 132.7369843, -204.5850677, 132.7100830, -337.2120667, 337.3220520
3: -216.7909698, 114.5724945, -216.7991943, 114.5775452, -331.3685303, 331.3717041
4: -199.6049042, 152.4937897, -199.7876434, 152.4877167, -352.0926208, 352.2814331
5: -177.9606171, 138.1950226, -178.0247498, 138.2140503, -316.1746521, 316.2197876
6: -170.9030914, 164.4172516, -170.9170990, 164.5043335, -335.4072876, 335.3343201
7: -185.8484650, 155.9080811, -185.9240417, 155.9380188, -341.7864990, 341.8320923
8: -224.2590790, 153.9212036, -224.2993927, 153.9011536, -378.1602173, 378.2205811
9: -169.2338867, 166.6506805, -169.3041687, 166.7497253, -335.9836121, 335.9548035

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2178051, upper bound: 372.2237338
time: 8.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2164827, upper bound: 372.2201506
time: 9.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -185.3669128, 147.4334106, -185.8121796, 147.7640839, -333.1309814, 333.2456055
1: -155.4396210, 130.6176758, -155.6785431, 130.8092804, -286.2488708, 286.2962036
2: -204.5019836, 132.7369843, -204.9757080, 132.8813629, -337.3833618, 337.7127075
3: -216.7909698, 114.5724945, -217.1641235, 114.7137756, -331.5047607, 331.7366333
4: -199.6049042, 152.4937897, -200.2001648, 152.6950684, -352.2999878, 352.6939087
5: -177.9606171, 138.1950226, -178.2984619, 138.3955231, -316.3561401, 316.4934692
6: -170.9030914, 164.4172516, -171.2279205, 164.8232574, -335.7262573, 335.6451416
7: -185.8484650, 155.9080811, -186.2083893, 156.1696930, -342.0181580, 342.1164551
8: -224.2590790, 153.9212036, -224.7252350, 154.1215820, -378.3806763, 378.6464233
9: -169.2338867, 166.6506805, -169.5990448, 167.0780334, -336.3119202, 336.2497253

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2178051, upper bound: 372.2237338
time: 8.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2164827, upper bound: 372.2201506
time: 8.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -184.3007050, 146.5994263, -178.5524292, 141.9743042, -326.2749329, 325.1518250
1: -154.4527893, 129.8132324, -149.5860443, 125.6986771, -280.1514587, 279.3992920
2: -203.3038940, 131.8618774, -196.9370270, 127.6436768, -330.9475708, 328.7988281
3: -215.4766388, 113.8078079, -208.6541748, 110.2710419, -325.7476501, 322.4619446
4: -198.4796143, 151.5153351, -192.3668976, 146.7283936, -345.2080078, 343.8822327
5: -176.8555450, 137.3049164, -171.3526764, 133.0219879, -309.8775024, 308.6575928
6: -169.8852234, 163.4695129, -164.5363770, 158.3761749, -328.2614136, 328.0058899
7: -184.6803589, 154.9273529, -178.9401855, 150.0865479, -334.7668762, 333.8674927
8: -222.9584503, 152.9478607, -215.9549103, 148.0840149, -371.0424500, 368.9027710
9: -168.2257996, 165.6989288, -162.9889069, 160.5521393, -328.7779236, 328.6878357

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 133

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1983740, upper bound: 372.1970730
time: 8.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1978513, upper bound: 372.1939156
time: 9.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.42 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2173816, upper bound: 372.2203715
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2154256, upper bound: 372.2170443
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2173816, upper bound: 372.2203715
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2154256, upper bound: 372.2170443
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2154357, upper bound: 372.2198634
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2139083, upper bound: 372.2166851
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2154357, upper bound: 372.2198634
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2139083, upper bound: 372.2166851
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2197532, upper bound: 372.2241287
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2180359, upper bound: 372.2203429
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2197532, upper bound: 372.2241287
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2180359, upper bound: 372.2203429
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.1993581, upper bound: 372.1974775
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.1988048, upper bound: 372.1941774
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2178051, upper bound: 372.2237338
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2164827, upper bound: 372.2201506
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2178051, upper bound: 372.2237338
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.2164827, upper bound: 372.2201506
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.1983740, upper bound: 372.1970730
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 18.42
Output dim: 2, lower bound: -372.1978513, upper bound: 372.1939156

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -176.0390015, 140.0690460, -181.9089813, 144.6808624, -320.7198486, 321.9780273
1: -147.5320587, 124.0483627, -152.5251923, 128.1522064, -275.6842651, 276.5735474
2: -194.1910400, 125.9824677, -200.7107391, 130.2316284, -324.4226685, 326.6932068
3: -205.8412018, 108.7975693, -212.7148285, 112.4376526, -318.2787781, 321.5123596
4: -189.5273743, 144.7961578, -195.9574738, 149.6167450, -339.1440735, 340.7536316
5: -168.9959564, 131.2100830, -174.6291962, 135.6066284, -304.6025696, 305.8392944
6: -162.2941132, 156.1309052, -167.6754913, 161.3809814, -323.6750793, 323.8063965
7: -176.4303436, 148.0275421, -182.4007416, 152.9992676, -329.4296265, 330.4282227
8: -213.0091095, 146.1764679, -220.0738220, 151.0556641, -364.0646973, 366.2502441
9: -160.6761169, 158.2586670, -166.0917206, 163.5734253, -324.2495422, 324.3504028

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2116633, upper bound: 372.2082220
time: 9.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2049547, upper bound: 372.2058632
time: 8.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -176.0390015, 140.0690460, -182.2918854, 144.9924316, -321.0314331, 322.3608704
1: -147.5320587, 124.0483627, -152.7680511, 128.3775940, -275.9096680, 276.8164062
2: -194.1910400, 125.9824677, -201.1207886, 130.4113922, -324.6024170, 327.1032410
3: -205.8412018, 108.7975693, -213.1065826, 112.5796204, -318.4208069, 321.9041443
4: -189.5273743, 144.7961578, -196.3878326, 149.8419800, -339.3693237, 341.1839600
5: -168.9959564, 131.2100830, -174.9210205, 135.8021393, -304.7980957, 306.1311035
6: -162.2941132, 156.1309052, -168.0033417, 161.7181702, -324.0122681, 324.1341858
7: -176.4303436, 148.0275421, -182.6997223, 153.2465210, -329.6768494, 330.7272034
8: -213.0091095, 146.1764679, -220.5234833, 151.2899933, -364.2990723, 366.6999207
9: -160.6761169, 158.2586670, -166.4085846, 163.9191589, -324.5952454, 324.6672363

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 133

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1973427, upper bound: 372.1979676
time: 11.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1949564, upper bound: 372.1976706
time: 8.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -172.7563477, 137.4644928, -182.7897491, 145.3648224, -318.1211548, 320.2542419
1: -144.7397766, 121.7224808, -153.1844177, 128.7233276, -273.4631042, 274.9068909
2: -190.5704193, 123.6329498, -201.6471558, 130.7941742, -321.3645935, 325.2800903
3: -201.9772186, 106.7602310, -213.6770630, 112.9289246, -314.9061279, 320.4372864
4: -186.0331573, 142.0839233, -196.9319305, 150.2973328, -336.3305054, 339.0157776
5: -165.8473816, 128.7656860, -175.4741058, 136.2246704, -302.0720520, 304.2398071
6: -159.2639618, 153.2241364, -168.4648132, 162.1426392, -321.4066162, 321.6889343
7: -173.1429596, 145.2635956, -183.2456665, 153.6946564, -326.8376160, 328.5092773
8: -209.0292053, 143.4426117, -221.0813293, 151.6914520, -360.7206421, 364.5239258
9: -157.6779327, 155.3111725, -166.8652191, 164.3512268, -322.0291443, 322.1763916

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2092788, upper bound: 372.2070504
time: 8.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2019534, upper bound: 372.2047274
time: 8.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -172.7563477, 137.4644928, -183.1268768, 145.6411896, -318.3975220, 320.5913696
1: -144.7397766, 121.7224808, -153.3924255, 128.9164886, -273.6562500, 275.1148987
2: -190.5704193, 123.6329498, -202.0088654, 130.9500732, -321.5205078, 325.6418152
3: -201.9772186, 106.7602310, -214.0114441, 113.0498123, -315.0270081, 320.7716675
4: -186.0331573, 142.0839233, -197.3177185, 150.4851074, -336.5182495, 339.4015808
5: -165.8473816, 128.7656860, -175.7221222, 136.3880615, -302.2354431, 304.4877930
6: -159.2639618, 153.2241364, -168.7533722, 162.4378967, -321.7018433, 321.9774780
7: -173.1429596, 145.2635956, -183.5079193, 153.9051208, -327.0480652, 328.7715149
8: -209.0292053, 143.4426117, -221.4786072, 151.8947906, -360.9239197, 364.9211426
9: -157.6779327, 155.3111725, -167.1368561, 164.6577148, -322.3356323, 322.4480286

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 205

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1953958, upper bound: 372.1971042
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1927464, upper bound: 372.1969062
time: 8.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -180.3045044, 143.4183960, -182.9610138, 145.5053864, -325.8098450, 326.3793945
1: -151.1448517, 127.0371170, -153.4241943, 128.8901367, -280.0349731, 280.4613037
2: -198.8913269, 129.0002289, -201.8708649, 130.9758148, -329.8671265, 330.8710632
3: -210.8210754, 111.4098816, -213.9420013, 113.0838470, -323.9049072, 325.3518677
4: -194.1367493, 148.2768402, -197.0878143, 150.4709015, -344.6076660, 345.3646240
5: -173.0574646, 134.3720398, -175.6292267, 136.3879395, -309.4454041, 310.0011597
6: -166.2186890, 159.9192200, -168.6406250, 162.3180542, -328.5367432, 328.5597839
7: -180.6997375, 151.5997467, -183.4545288, 153.8813934, -334.5811157, 335.0542603
8: -218.1634674, 149.6729889, -221.3483124, 151.9180145, -370.0814209, 371.0213013
9: -164.5925903, 162.1057739, -167.0601196, 164.5274353, -329.1200256, 329.1658936

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2129651, upper bound: 372.2106673
time: 8.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2066977, upper bound: 372.2084498
time: 8.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -178.3540802, 141.8708496, -178.4544678, 141.9356689, -320.2897339, 320.3252869
1: -149.4317322, 125.6361771, -149.5977631, 125.7064896, -275.1381836, 275.2339478
2: -196.7313385, 127.5147324, -196.8964539, 127.7065582, -324.4378967, 324.4111328
3: -208.4779510, 110.1315613, -208.6313477, 110.2777710, -318.7557373, 318.7628479
4: -192.0453491, 146.6000824, -192.2463837, 146.7361908, -338.7815552, 338.8464661
5: -171.1473236, 132.8709717, -171.2831116, 133.0121765, -304.1594849, 304.1539917
6: -164.3778381, 158.1772156, -164.4694366, 158.3234711, -322.7012634, 322.6466370
7: -178.7006531, 149.9170837, -178.9086914, 150.0709534, -328.7715759, 328.8257446
8: -215.8012085, 147.9423523, -215.9078217, 148.1590271, -363.9601746, 363.8501282
9: -162.7856140, 160.3403320, -162.9435120, 160.4811859, -323.2667847, 323.2838440

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2110178, upper bound: 372.2049599
time: 7.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2045718, upper bound: 372.2027923
time: 5.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.62 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2116633, upper bound: 372.2082220
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2049547, upper bound: 372.2058632
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.1973427, upper bound: 372.1979676
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.1949564, upper bound: 372.1976706
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2092788, upper bound: 372.2070504
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2019534, upper bound: 372.2047274
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.1953958, upper bound: 372.1971042
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.1927464, upper bound: 372.1969062
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2129651, upper bound: 372.2106673
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2066977, upper bound: 372.2084498
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2110178, upper bound: 372.2049599
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.62
Output dim: 2, lower bound: -372.2045718, upper bound: 372.2027923
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.62
Output dim: 2, lower bound: -372.2197532, upper bound: 372.2241287
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.62
Output dim: 2, lower bound: -372.2180359, upper bound: 372.2203429
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.62
Output dim: 2, lower bound: -372.2178051, upper bound: 372.2237338
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.62
Output dim: 2, lower bound: -372.2164827, upper bound: 372.2201506
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.62
Output dim: 2, lower bound: -372.2178051, upper bound: 372.2237338
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.62
Output dim: 2, lower bound: -372.2164827, upper bound: 372.2201506
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=375.07427978515625
rel_dist={2: [-372.2698094205705, 372.2698094230649]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2674396, upper bound: 372.2671197
time: 15.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2686319, upper bound: 372.2686319
time: 13.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 29.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 29.57
Output dim: 2, lower bound: -372.2674396, upper bound: 372.2671197
IS_A2, status: Status.UNKNOWN, split count: 1, time: 29.57
Output dim: 2, lower bound: -372.2686319, upper bound: 372.2686319

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -195.3206635, 155.3128815, -199.2161865, 158.3800049, -353.7006836, 354.5290222
1: -163.8407288, 137.6345367, -167.1292877, 140.3650818, -304.2057495, 304.7638245
2: -215.4388123, 139.8561096, -219.7322083, 142.6119537, -358.0506592, 359.5882874
3: -228.5296783, 120.7476654, -233.0905151, 123.1332397, -351.6629028, 353.8381958
4: -210.1535492, 160.6972504, -214.3579407, 163.8820496, -374.0355530, 375.0551147
5: -187.5573120, 145.6229858, -191.2771759, 148.5067902, -336.0640869, 336.9001465
6: -180.0850830, 173.1997681, -183.6669464, 176.6582184, -356.7432861, 356.8666992
7: -195.8436737, 164.3047943, -199.7415924, 167.5682068, -363.4118652, 364.0463562
8: -236.2275085, 162.1352081, -240.9219055, 165.3315125, -401.5589905, 403.0571289
9: -178.2790680, 175.5395508, -181.8448639, 179.0483551, -357.3273926, 357.3843994

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2554944, upper bound: 372.2549564
time: 13.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2559902, upper bound: 372.2556128
time: 15.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -199.6509399, 158.7141571, -201.3076019, 160.0225677, -359.6735229, 360.0217590
1: -167.5093231, 140.6702576, -168.9146576, 141.8318481, -309.3411560, 309.5848999
2: -220.2118988, 142.9197845, -222.0383606, 144.0907593, -364.3026123, 364.9581299
3: -233.5867157, 123.4031067, -235.5299225, 124.4180832, -358.0047913, 358.9329834
4: -214.8302155, 164.2307281, -216.6023865, 165.5814819, -380.4116821, 380.8330994
5: -191.6817169, 148.8310699, -193.2676086, 150.0583801, -341.7400513, 342.0986633
6: -184.0677643, 177.0479431, -185.5851593, 178.5199127, -362.5876770, 362.6331177
7: -200.1786194, 167.9325256, -201.8345795, 169.3208923, -369.4995117, 369.7670898
8: -241.4618378, 165.6829529, -243.4563141, 167.0451508, -408.5069885, 409.1392517
9: -182.2528229, 179.4474182, -183.7667542, 180.9452820, -363.1980286, 363.2141724

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2581568, upper bound: 372.2579985
time: 14.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2584661, upper bound: 372.2584661
time: 12.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.05 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.05
Output dim: 2, lower bound: -372.2554944, upper bound: 372.2549564
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.05
Output dim: 2, lower bound: -372.2559902, upper bound: 372.2556128
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.05
Output dim: 2, lower bound: -372.2581568, upper bound: 372.2579985
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.05
Output dim: 2, lower bound: -372.2584661, upper bound: 372.2584661

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -187.1559906, 148.8294220, -187.8349152, 149.3418274, -336.4978027, 336.6643372
1: -156.9340210, 131.8548279, -157.5007935, 132.3084564, -289.2424927, 289.3555908
2: -206.4622192, 134.0307922, -207.2196808, 134.4916229, -340.9537964, 341.2503357
3: -218.9193878, 115.7223206, -219.6948242, 116.1274414, -335.0468140, 335.4171448
4: -201.4987335, 153.9686584, -202.2945709, 154.5022430, -356.0009766, 356.2631531
5: -179.7162781, 139.5532379, -180.3471680, 140.0454254, -319.7616577, 319.9003906
6: -172.5387115, 165.9913025, -173.1475525, 166.6103058, -339.1490173, 339.1388550
7: -187.7036591, 157.4531097, -188.3952332, 158.0168457, -345.7204285, 345.8483276
8: -226.3645935, 155.3848267, -227.1735229, 155.9219055, -382.2864685, 382.5583496
9: -170.8472595, 168.2214661, -171.4857788, 168.8468018, -339.6940308, 339.7072449

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2375086, upper bound: 372.2376274
time: 15.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2369340, upper bound: 372.2364330
time: 11.42 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -183.6215820, 146.0281830, -188.6757965, 149.9947968, -333.6163940, 334.7039490
1: -153.9147339, 129.3504639, -158.1245728, 132.8523254, -286.7669983, 287.4750061
2: -202.5565491, 131.4942474, -208.1106567, 135.0258942, -337.5824585, 339.6048889
3: -214.7584839, 113.5184555, -220.6085663, 116.5957565, -331.3542480, 334.1270142
4: -197.7281952, 151.0430756, -203.2279358, 155.1485291, -352.8767090, 354.2709961
5: -176.3305054, 136.9167480, -181.1540985, 140.6343689, -316.9648132, 318.0708618
6: -169.2778778, 162.8579254, -173.8985901, 167.3364868, -336.6143494, 336.7565308
7: -184.1525574, 154.4735413, -189.2004700, 158.6808624, -342.8334045, 343.6739502
8: -222.0708160, 152.4245300, -228.1285858, 156.5232391, -378.5940552, 380.5530701
9: -167.6147766, 165.0475922, -172.2233582, 169.5875549, -337.2022705, 337.2709351

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2372865, upper bound: 372.2381778
time: 15.06 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2374609, upper bound: 372.2370096
time: 11.97 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -191.4287415, 152.1849060, -189.8500671, 150.9244843, -342.3532104, 342.0349426
1: -160.5547638, 134.8500671, -159.2225647, 133.7217560, -294.2765198, 294.0726013
2: -211.1716919, 137.0547180, -209.4420776, 135.9169464, -347.0886230, 346.4967957
3: -223.9112549, 118.3430710, -222.0462799, 117.3664703, -341.2777100, 340.3893127
4: -206.1158295, 157.4542847, -204.4590302, 156.1385956, -362.2544250, 361.9132996
5: -183.7863312, 142.7192383, -182.2656555, 141.5409698, -325.3273010, 324.9848938
6: -176.4684143, 169.7895050, -174.9956818, 168.4055023, -344.8738403, 344.7851257
7: -191.9828949, 161.0334015, -190.4135284, 159.7066193, -351.6894531, 351.4469299
8: -231.5276489, 158.8857727, -229.6142731, 157.5737305, -389.1013184, 388.5000305
9: -174.7707367, 172.0764771, -173.3400726, 170.6739960, -345.4447327, 345.4164734

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2401500, upper bound: 372.2407197
time: 11.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2397252, upper bound: 372.2396210
time: 10.38 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -188.0335846, 149.4956970, -190.8421783, 151.6987305, -339.7322998, 340.3378906
1: -157.6543732, 132.4429626, -159.9760590, 134.3713074, -292.0256958, 292.4189453
2: -207.4205322, 134.6170959, -210.5003662, 136.5587921, -343.9793091, 345.1174622
3: -219.9115906, 116.2255173, -223.1368561, 117.9279633, -337.8395386, 339.3623657
4: -202.4922028, 154.6457825, -205.5523376, 156.9115601, -359.4036865, 360.1980896
5: -180.5316162, 140.1876068, -183.2151794, 142.2441101, -322.7757263, 323.4027710
6: -173.3389740, 166.7764435, -175.8894806, 169.2627258, -342.6016846, 342.6659241
7: -188.5692444, 158.1684570, -191.3686523, 160.4955292, -349.0647583, 349.5371094
8: -227.4075165, 156.0456238, -230.7584839, 158.3042450, -385.7117310, 386.8040466
9: -171.6632080, 169.0288696, -174.2140045, 171.5532532, -343.2164612, 343.2428284

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2405181, upper bound: 372.2411975
time: 11.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2395000, upper bound: 372.2401248
time: 13.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2375086, upper bound: 372.2376274
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2369340, upper bound: 372.2364330
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2372865, upper bound: 372.2381778
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2374609, upper bound: 372.2370096
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2401500, upper bound: 372.2407197
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2397252, upper bound: 372.2396210
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2405181, upper bound: 372.2411975
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.44
Output dim: 2, lower bound: -372.2395000, upper bound: 372.2401248

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -179.4786377, 142.7946472, -182.9086304, 145.4691315, -324.9477234, 325.7032471
1: -150.4676208, 126.4808807, -153.3510132, 128.8605194, -279.3281250, 279.8319092
2: -198.0261688, 128.5679321, -201.8062439, 130.9866486, -329.0128174, 330.3741760
3: -209.9005432, 110.9738770, -213.9068604, 113.0805588, -322.9810791, 324.8807373
4: -193.2889099, 147.6824341, -197.0266266, 150.4687195, -343.7576294, 344.7090454
5: -172.3399506, 133.8314819, -175.6134186, 136.3740387, -308.7139587, 309.4448853
6: -165.4737701, 159.2017822, -168.6139984, 162.2533722, -327.7271423, 327.8157959
7: -179.9740448, 150.9816132, -183.4354706, 153.8641205, -333.8381653, 334.4170532
8: -217.1526642, 149.1033020, -221.2623749, 151.8911591, -369.0437317, 370.3656616
9: -163.8484955, 161.3506775, -166.9952850, 164.4374847, -328.2859802, 328.3458862

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2149083, upper bound: 372.2144775
time: 13.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2123987, upper bound: 372.2130296
time: 11.38 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -179.9633179, 143.1827545, -175.9212036, 139.9482117, -319.9115295, 319.1039429
1: -150.7948608, 126.7771988, -147.4684448, 123.9477997, -274.7426758, 274.2456360
2: -198.5465546, 128.8216705, -194.1125183, 125.9804764, -324.5270386, 322.9342041
3: -210.4092560, 111.1775589, -205.6791382, 108.7500381, -319.1593018, 316.8566589
4: -193.8309479, 147.9904480, -189.5367279, 144.7170715, -338.5479736, 337.5271606
5: -172.7248077, 134.1036987, -168.8766174, 131.1586609, -303.8834839, 302.9803162
6: -165.8941650, 159.6287079, -162.1850433, 156.0685272, -321.9626770, 321.8137512
7: -180.3762817, 151.3154602, -176.3928986, 147.9704742, -328.3467407, 327.7083740
8: -217.7235260, 149.4207916, -212.8610687, 146.1287689, -363.8522339, 362.2818604
9: -164.2602844, 161.7862244, -160.6288452, 158.1774902, -322.4377747, 322.4150085

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2135917, upper bound: 372.2115821
time: 12.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2116397, upper bound: 372.2109308
time: 11.68 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.8959961, 139.9539490, -183.7373505, 146.1117706, -322.0077515, 323.6912537
1: -147.4075623, 123.9414825, -153.9648132, 129.3942719, -276.8017883, 277.9063110
2: -194.0679016, 125.9962845, -202.6847382, 131.5108185, -325.5786438, 328.6810303
3: -205.6835938, 108.7383194, -214.8075562, 113.5398560, -319.2234497, 323.5458374
4: -189.4654083, 144.7181549, -197.9461975, 151.1052399, -340.5706482, 342.6643677
5: -168.9067535, 131.1588898, -176.4083099, 136.9529114, -305.8596802, 307.5671692
6: -162.1691589, 156.0254974, -169.3542023, 162.9689789, -325.1381226, 325.3796692
7: -176.3747864, 147.9595032, -184.2282867, 154.5169067, -330.8916321, 332.1878052
8: -212.8012085, 146.1035767, -222.2034149, 152.4828796, -365.2840881, 368.3070068
9: -160.5711517, 158.1326294, -167.7213898, 165.1678772, -325.7389832, 325.8540039

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2149920, upper bound: 372.2144705
time: 11.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2118279, upper bound: 372.2129214
time: 10.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -176.3816071, 140.3432617, -176.8829651, 140.6973267, -317.0788574, 317.2261047
1: -147.7371216, 124.2354279, -148.1967010, 124.5778809, -272.3150024, 272.4321289
2: -194.5885620, 126.2571335, -195.1391754, 126.6048431, -321.1934204, 321.3963013
3: -206.1841278, 108.9473267, -206.7377167, 109.2941360, -315.4782715, 315.6850586
4: -190.0083313, 145.0249939, -190.6021881, 145.4640350, -335.4723511, 335.6271667
5: -169.2905426, 131.4331665, -169.8012543, 131.8378448, -301.1283875, 301.2344360
6: -162.5893097, 156.4498749, -163.0494385, 156.9017944, -319.4910583, 319.4992981
7: -176.7840576, 148.2938690, -177.3251648, 148.7380676, -325.5221252, 325.6190186
8: -213.3724518, 146.4250946, -213.9661255, 146.8342743, -360.2066956, 360.3912354
9: -160.9788055, 158.5700684, -161.4804840, 159.0311890, -320.0099487, 320.0505371

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2134498, upper bound: 372.2115310
time: 14.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2111881, upper bound: 372.2108145
time: 11.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -183.7161255, 146.1193542, -184.9116058, 147.0400543, -330.7561340, 331.0309143
1: -154.0586853, 129.4515533, -155.0631256, 130.2648773, -284.3235474, 284.5146790
2: -202.6967010, 131.5677490, -204.0149841, 132.4030151, -335.0997314, 335.5827332
3: -214.8511658, 113.5718918, -216.2437897, 114.3112640, -329.1623535, 329.8156433
4: -197.8692627, 151.1395874, -199.1779480, 152.0948029, -349.9640198, 350.3175354
5: -176.3731995, 136.9727020, -177.5182343, 137.8608398, -314.2340088, 314.4909058
6: -169.3724518, 162.9680176, -170.4515839, 164.0374756, -333.4099121, 333.4196167
7: -184.2183990, 154.5330811, -185.4410858, 155.5438080, -339.7622070, 339.9741211
8: -222.2748871, 152.5746460, -223.6888885, 153.5327911, -375.8076782, 376.2634888
9: -167.7417145, 165.1740417, -168.8386688, 166.2537537, -333.9954224, 334.0126953

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2184838, upper bound: 372.2182291
time: 12.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2123987, upper bound: 372.2167266
time: 13.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -184.0533447, 146.4024353, -177.9595795, 141.5522003, -325.6054688, 324.3619995
1: -154.2651215, 129.6412811, -149.2115326, 125.3768082, -279.6419373, 278.8528137
2: -203.0541229, 131.7164307, -196.3617554, 127.4201050, -330.4741821, 328.0781860
3: -215.1911011, 113.6835861, -208.0596619, 110.0038452, -325.1949158, 321.7432556
4: -198.2505341, 151.3292389, -191.7272491, 146.3731537, -344.6236877, 343.0564880
5: -176.6240692, 137.1393127, -170.8193512, 132.6734161, -309.2974548, 307.9586792
6: -169.6552124, 163.2673035, -164.0555878, 157.8845367, -327.5397034, 327.3228760
7: -184.4689636, 154.7425690, -178.4336548, 149.6801147, -334.1489868, 333.1760559
8: -222.6729736, 152.7759399, -215.3307648, 147.7994232, -370.4724121, 368.1066895
9: -168.0181885, 165.4828033, -162.5063324, 160.0287476, -328.0469360, 327.9891357

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2172141, upper bound: 372.2152956
time: 14.02 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2150504, upper bound: 372.2146217
time: 12.10 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -180.2735443, 143.3936157, -185.8862762, 147.8018036, -328.0753479, 329.2798767
1: -151.1185150, 127.0107651, -155.8026581, 130.9020233, -282.0205383, 282.8134155
2: -198.8937988, 129.0966339, -205.0554657, 133.0323639, -331.9261169, 334.1520386
3: -210.7965393, 111.4243774, -217.3157501, 114.8619080, -325.6584473, 328.7401123
4: -194.1945038, 148.2928162, -200.2538757, 152.8537445, -347.0482483, 348.5466309
5: -173.0732574, 134.4050293, -178.4520111, 138.5506439, -311.6239014, 312.8570557
6: -166.1985168, 159.9138947, -171.3295746, 164.8802643, -331.0787354, 331.2434082
7: -180.7585144, 151.6273804, -186.3802032, 156.3178711, -337.0762634, 338.0074768
8: -218.0972443, 149.6965027, -224.8133392, 154.2500000, -372.3472290, 374.5098267
9: -164.5904694, 162.0832214, -169.6973724, 167.1179199, -331.7083740, 331.7805786

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2187034, upper bound: 372.2183678
time: 11.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2152850, upper bound: 372.2167918
time: 11.92 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -180.6481171, 143.6999817, -179.0955353, 142.4367828, -323.0848694, 322.7955322
1: -151.3560944, 127.2242737, -150.0869446, 126.1283035, -277.4844055, 277.3111877
2: -199.2901001, 129.2746277, -197.5790558, 128.1671600, -327.4572449, 326.8536987
3: -211.1716156, 111.5610123, -209.3194580, 110.6536942, -321.8252869, 320.8804626
4: -194.6126251, 148.5073853, -192.9755859, 147.2632294, -341.8758545, 341.4829712
5: -173.3554382, 134.5968475, -171.9067078, 133.4813690, -306.8367920, 306.5035400
6: -166.5112000, 160.2434387, -165.0797272, 158.8699036, -325.3811035, 325.3231812
7: -181.0470734, 151.8665466, -179.5347137, 150.5907898, -331.6378479, 331.4011841
8: -218.5331268, 149.9245758, -216.6475983, 148.6487579, -367.1818848, 366.5721741
9: -164.8937073, 162.4242859, -163.5134125, 161.0379791, -325.9316711, 325.9376221

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2172122, upper bound: 372.2153356
time: 12.28 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2146067, upper bound: 372.2146067
time: 8.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.97 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2149083, upper bound: 372.2144775
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2123987, upper bound: 372.2130296
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2135917, upper bound: 372.2115821
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2116397, upper bound: 372.2109308
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2149920, upper bound: 372.2144705
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2118279, upper bound: 372.2129214
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2134498, upper bound: 372.2115310
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2111881, upper bound: 372.2108145
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2184838, upper bound: 372.2182291
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2123987, upper bound: 372.2167266
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2172141, upper bound: 372.2152956
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2150504, upper bound: 372.2146217
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2187034, upper bound: 372.2183678
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2152850, upper bound: 372.2167918
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2172122, upper bound: 372.2153356
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 21.97
Output dim: 2, lower bound: -372.2146067, upper bound: 372.2146067
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=375.07427978515625
rel_dist={2: [-372.26978283690505, 372.26978280960407]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2676862, upper bound: 372.2672150
time: 9.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2686804, upper bound: 372.2686804
time: 9.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.26
Output dim: 2, lower bound: -372.2676862, upper bound: 372.2672150
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.26
Output dim: 2, lower bound: -372.2686804, upper bound: 372.2686804

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -195.3206635, 155.3128815, -201.0755005, 159.8465271, -355.1671753, 356.3883362
1: -163.8407288, 137.6345367, -168.7098846, 141.6687469, -305.5094604, 306.3444214
2: -215.4388123, 139.8561096, -221.7829285, 143.9259796, -359.3646851, 361.6390076
3: -228.5296783, 120.7476654, -235.2677002, 124.2732544, -352.8029175, 356.0153809
4: -210.1535492, 160.6972504, -216.3494263, 165.3961945, -375.5497437, 377.0466919
5: -187.5573120, 145.6229858, -193.0541534, 149.8847351, -337.4420471, 338.6771240
6: -180.0850830, 173.1997681, -185.3709869, 178.3111420, -358.3961792, 358.5707397
7: -195.8436737, 164.3047943, -201.6009674, 169.1264191, -364.9700012, 365.9057007
8: -236.2275085, 162.1352081, -243.1651917, 166.8588867, -403.0863953, 405.3004150
9: -178.2790680, 175.5395508, -183.5468903, 180.7315979, -359.0106506, 359.0864258

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2562312, upper bound: 372.2552086
time: 9.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2563809, upper bound: 372.2557582
time: 10.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -199.6509399, 158.7141571, -202.6150818, 161.0551453, -360.7060852, 361.3292236
1: -167.5093231, 140.6702576, -170.0238342, 142.7483063, -310.2575378, 310.6940308
2: -220.2118988, 142.9197845, -223.4799194, 145.0145569, -365.2264404, 366.3996887
3: -233.5867157, 123.4031067, -237.0628815, 125.2191391, -358.8058472, 360.4659729
4: -214.8302155, 164.2307281, -218.0012512, 166.6471710, -381.4773865, 382.2319946
5: -191.6817169, 148.8310699, -194.5189667, 151.0267792, -342.7084961, 343.3500061
6: -184.0677643, 177.0479431, -186.7826996, 179.6815033, -363.7492371, 363.8306274
7: -200.1786194, 167.9325256, -203.1415558, 170.4164886, -370.5950928, 371.0740967
8: -241.4618378, 165.6829529, -245.0304260, 168.1197205, -409.5815125, 410.7133789
9: -182.2528229, 179.4474182, -184.9613647, 182.1275024, -364.3802185, 364.4086914

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2584707, upper bound: 372.2581630
time: 9.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2585651, upper bound: 372.2585651
time: 8.12 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.94 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.94
Output dim: 2, lower bound: -372.2562312, upper bound: 372.2552086
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.94
Output dim: 2, lower bound: -372.2563809, upper bound: 372.2557582
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.94
Output dim: 2, lower bound: -372.2584707, upper bound: 372.2581630
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.94
Output dim: 2, lower bound: -372.2585651, upper bound: 372.2585651

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -189.8300323, 150.9526215, -189.6793213, 150.7966003, -340.6266479, 340.6318970
1: -159.1963196, 133.7477417, -159.0691528, 133.6017761, -292.7980957, 292.8168945
2: -209.4019470, 135.9386597, -209.2539825, 135.7953949, -345.1973267, 345.1926270
3: -222.0664520, 117.3685837, -221.8546143, 117.2585602, -339.3250122, 339.2231750
4: -204.3328705, 156.1722717, -204.2701111, 156.0041962, -360.3370667, 360.4423828
5: -182.2840271, 141.5413818, -182.1098633, 141.4122925, -323.6963196, 323.6512451
6: -175.0102539, 168.3520966, -174.8381348, 168.2500305, -343.2601929, 343.1901855
7: -190.3693390, 159.6973267, -190.2397766, 159.5627136, -349.9320679, 349.9371033
8: -229.5944519, 157.5953827, -229.3985291, 157.4373932, -387.0317688, 386.9938965
9: -173.2813416, 170.6182098, -173.1743011, 170.5165405, -343.7978210, 343.7925110

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2386302, upper bound: 372.2388070
time: 10.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2375871, upper bound: 372.2366952
time: 12.34 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -186.5245819, 148.3312073, -190.5326080, 151.4590912, -337.9836731, 338.8638306
1: -156.3780823, 131.4057922, -159.7032318, 134.1544952, -290.5325623, 291.1090088
2: -205.7528534, 133.5691223, -210.1582184, 136.3386383, -342.0914917, 343.7272949
3: -218.1750641, 115.3130112, -222.7825470, 117.7343597, -335.9094238, 338.0955200
4: -200.8112030, 153.4378204, -205.2164001, 156.6607819, -357.4719849, 358.6542358
5: -179.1151581, 139.0769348, -182.9283905, 142.0108337, -321.1259766, 322.0053101
6: -171.9588928, 165.4240265, -175.6004791, 168.9868317, -340.9456482, 341.0244751
7: -187.0534821, 156.9131165, -191.0571289, 160.2369690, -347.2904663, 347.9701233
8: -225.5827942, 154.8343811, -230.3691559, 158.0490723, -383.6318359, 385.2035217
9: -170.2602081, 167.6503754, -173.9230347, 171.2681122, -341.5282593, 341.5734253

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2389387, upper bound: 372.2392200
time: 8.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2379325, upper bound: 372.2371331
time: 8.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -194.1250000, 154.3258209, -191.1632690, 151.9615173, -346.0865173, 345.4890747
1: -162.8355103, 136.7585449, -160.3367004, 134.6424713, -297.4779663, 297.0951843
2: -214.1356659, 138.9781952, -210.8904114, 136.8446960, -350.9802856, 349.8685608
3: -227.0840149, 120.0028076, -223.5861664, 118.1708527, -345.2548828, 343.5888977
4: -208.9732819, 159.6763763, -205.8639221, 157.2089539, -366.1822205, 365.5402832
5: -186.3753662, 144.7234802, -183.5220184, 142.5137939, -328.8890991, 328.2454529
6: -178.9600525, 172.1696014, -176.1985474, 169.5721741, -348.5322266, 348.3681641
7: -194.6705475, 163.2959137, -191.7260437, 160.8067932, -355.4773254, 355.0219727
8: -234.7848511, 161.1143951, -231.1955414, 158.6533508, -393.4382019, 392.3099365
9: -177.2241669, 174.4935455, -174.5398865, 171.8616333, -349.0858154, 349.0334473

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2407638, upper bound: 372.2418113
time: 11.12 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2399645, upper bound: 372.2397718
time: 8.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -190.9159546, 151.7822418, -192.1278687, 152.7141418, -343.6300964, 343.9100647
1: -160.0995789, 134.4833069, -161.0666351, 135.2729340, -295.3724976, 295.5499268
2: -210.5943146, 136.6772919, -211.9179993, 137.4674683, -348.0617371, 348.5952759
3: -223.3036041, 118.0066376, -224.6448212, 118.7154465, -342.0189819, 342.6513672
4: -205.5536652, 157.0235901, -206.9282379, 157.9593048, -363.5129395, 363.9518433
5: -183.2971954, 142.3317871, -184.4459686, 143.1962433, -326.4934387, 326.7777100
6: -176.0008392, 169.3249969, -177.0667114, 170.4054718, -346.4063110, 346.3917236
7: -191.4497833, 160.5907288, -192.6537323, 161.5731659, -353.0229187, 353.2444458
8: -230.8943634, 158.4372101, -232.3063354, 159.3609619, -390.2553101, 390.7435303
9: -174.2900391, 171.6133270, -175.3885956, 172.7160339, -347.0060425, 347.0019226

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2409324, upper bound: 372.2421985
time: 8.74 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2401756, upper bound: 372.2401756
time: 7.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2386302, upper bound: 372.2388070
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2375871, upper bound: 372.2366952
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2389387, upper bound: 372.2392200
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2379325, upper bound: 372.2371331
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2407638, upper bound: 372.2418113
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2399645, upper bound: 372.2397718
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2409324, upper bound: 372.2421985
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.68
Output dim: 2, lower bound: -372.2401756, upper bound: 372.2401756

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -182.1353302, 144.9039612, -187.0901489, 148.7610168, -330.8963013, 331.9940796
1: -152.7153778, 128.3616180, -156.8884430, 131.7896271, -284.5050049, 285.2500305
2: -200.9469452, 130.4632721, -206.4088440, 133.9531555, -334.9000854, 336.8721313
3: -213.0276642, 112.6096039, -218.8127441, 115.6573334, -328.6849060, 331.4223633
4: -196.1044159, 149.8721008, -201.5016785, 153.8841705, -349.9885864, 351.3737793
5: -174.8906555, 135.8066254, -179.6216583, 139.4828949, -314.3734741, 315.4282837
6: -167.9295654, 161.5469818, -172.4556122, 165.9603271, -333.8898926, 334.0025940
7: -182.6222534, 153.2109680, -187.6332245, 157.3802490, -340.0025024, 340.8441772
8: -220.3620911, 151.2994080, -226.2923126, 155.3187256, -375.6807861, 377.5916748
9: -166.2668304, 163.7320099, -170.8144226, 168.1993103, -334.4660950, 334.5464172

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2184162, upper bound: 372.2177288
time: 8.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2140325, upper bound: 372.2151712
time: 10.73 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -182.6457825, 145.3117218, -180.9186249, 143.8891907, -326.5349731, 326.2303467
1: -153.0641327, 128.6758423, -151.6920929, 127.4532394, -280.5173645, 280.3679199
2: -201.4949646, 130.7361298, -199.6152344, 129.5357971, -331.0307617, 330.3513489
3: -213.5667419, 112.8289108, -211.5475006, 111.8327332, -325.3994446, 324.3763428
4: -196.6748047, 150.2011719, -194.8881989, 148.8090973, -345.4838867, 345.0893555
5: -175.3002319, 136.0984802, -173.6753693, 134.8764801, -310.1766968, 309.7738647
6: -168.3733521, 161.9964905, -166.7765198, 160.4978027, -328.8711548, 328.7730103
7: -183.0507660, 153.5667419, -181.4119263, 152.1744232, -335.2251892, 334.9786682
8: -220.9638062, 151.6387177, -218.8732910, 150.2375793, -371.2013855, 370.5119934
9: -166.7023163, 164.1901245, -165.1911163, 162.6707153, -329.3730469, 329.3812256

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2161206, upper bound: 372.2124516
time: 9.01 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2125272, upper bound: 372.2112188
time: 9.26 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -178.8022156, 142.2593689, -187.9384460, 149.4194031, -328.2215881, 330.1977844
1: -149.8737488, 125.9990387, -157.5184326, 132.3380585, -282.2117310, 283.5174561
2: -197.2676239, 128.0731506, -207.3081665, 134.4923553, -331.7599792, 335.3813171
3: -209.1033783, 110.5352707, -219.7352600, 116.1293793, -325.2327271, 330.2705383
4: -192.5521088, 147.1152649, -202.4424744, 154.5368958, -347.0889893, 349.5577393
5: -171.6944122, 133.3213501, -180.4355164, 140.0772400, -311.7716675, 313.7568054
6: -164.8535156, 158.5941772, -173.2137146, 166.6930695, -331.5465698, 331.8078918
7: -179.2786713, 150.4017181, -188.4452972, 158.0499725, -337.3286438, 338.8470154
8: -216.3173828, 148.5153503, -227.2573547, 155.9266510, -372.2439575, 375.7727051
9: -163.2198029, 160.7384033, -171.5586548, 168.9464417, -332.1662598, 332.2970276

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2185363, upper bound: 372.2176945
time: 10.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2128975, upper bound: 372.2149434
time: 9.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -179.3986511, 142.7367401, -181.8591003, 144.6211395, -324.0197449, 324.5958252
1: -150.2958984, 126.3735733, -152.4015961, 128.0685883, -278.3645020, 278.7751770
2: -197.9098053, 128.4116821, -200.6174164, 130.1445160, -328.0543213, 329.0290833
3: -209.7415924, 110.8093643, -212.5790863, 112.3646774, -322.1062622, 323.3883972
4: -193.2167664, 147.5171356, -195.9298553, 149.5376740, -342.7543945, 343.4469910
5: -172.1875916, 133.6794739, -174.5782928, 135.5404053, -307.7279968, 308.2577515
6: -165.3763733, 159.1186676, -167.6209717, 161.3121338, -326.6885071, 326.7396240
7: -179.7984314, 150.8318329, -182.3204498, 152.9237366, -332.7221069, 333.1522522
8: -217.0234680, 148.9304810, -219.9527588, 150.9233246, -367.9467773, 368.8832397
9: -163.7366180, 161.2760162, -166.0218964, 163.5042572, -327.2408752, 327.2979126

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2158053, upper bound: 372.2123259
time: 10.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2116797, upper bound: 372.2109743
time: 7.70 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -186.3916931, 148.2435455, -188.5712891, 149.9226074, -336.3143005, 336.8148193
1: -156.3221588, 131.3452148, -158.1534424, 132.8282776, -289.1503601, 289.4986572
2: -205.6378784, 133.4766998, -208.0419464, 135.0003815, -340.6382446, 341.5186462
3: -217.9996948, 115.2190018, -220.5400391, 116.5676041, -334.5672913, 335.7589722
4: -200.7048950, 153.3445435, -203.0922546, 155.0865631, -355.7914429, 356.4367981
5: -178.9417725, 138.9620361, -181.0302124, 140.5818634, -319.5236206, 319.9922180
6: -171.8454437, 165.3297272, -173.8135376, 167.2796478, -339.1250916, 339.1432190
7: -186.8854828, 156.7783203, -189.1162262, 158.6216888, -345.5071411, 345.8945312
8: -225.5071411, 154.7858124, -228.0856171, 156.5325775, -382.0397339, 382.8714294
9: -170.1766510, 167.5724640, -172.1771851, 169.5414581, -339.7180786, 339.7496338

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2219162, upper bound: 372.2213859
time: 10.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2169333, upper bound: 372.2186283
time: 9.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -186.7585754, 148.5492096, -182.4147491, 145.0665894, -331.8251648, 330.9639282
1: -156.5532379, 131.5557404, -152.9716187, 128.5026245, -285.0558472, 284.5273132
2: -206.0276642, 133.6467285, -201.2658844, 130.5935669, -336.6211853, 334.9125977
3: -218.3750153, 115.3486176, -213.2941895, 112.7535477, -331.1285706, 328.6427307
4: -201.1183319, 153.5581818, -196.4965363, 150.0244904, -351.1428223, 350.0547180
5: -179.2214508, 139.1502228, -175.1016846, 135.9885254, -315.2099609, 314.2518921
6: -172.1555634, 165.6548767, -168.1496124, 161.8312225, -333.9867859, 333.8044434
7: -187.1656952, 157.0124817, -182.9104309, 153.4300232, -340.5957031, 339.9229126
8: -225.9410706, 155.0122223, -220.6862640, 151.4633789, -377.4044495, 375.6984253
9: -170.4806061, 167.9071198, -166.5696106, 164.0289917, -334.5095825, 334.4767456

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2194976, upper bound: 372.2160248
time: 9.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2155830, upper bound: 372.2147986
time: 9.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -183.1604767, 145.6834564, -189.5270538, 150.6688690, -333.8293457, 335.2105103
1: -153.5675507, 129.0549469, -158.8763885, 133.4524689, -287.0200195, 287.9313354
2: -202.0725708, 131.1604004, -209.0605164, 135.6164551, -337.6890259, 340.2209167
3: -214.1937714, 113.2090378, -221.5891724, 117.1065826, -331.3003540, 334.7982178
4: -197.2612305, 150.6739197, -204.1473846, 155.8296814, -353.0909119, 354.8212585
5: -175.8434296, 136.5532074, -181.9462585, 141.2578278, -317.1012573, 318.4994507
6: -168.8650208, 162.4664917, -174.6735535, 168.1054230, -336.9704590, 337.1400452
7: -183.6436768, 154.0534973, -190.0354919, 159.3805084, -343.0241699, 344.0889893
8: -221.5899963, 152.0915985, -229.1859741, 157.2332764, -378.8231506, 381.2775574
9: -167.2221069, 164.6719208, -173.0183105, 170.3883820, -337.6104736, 337.6902466

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2220912, upper bound: 372.2214364
time: 9.13 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2159392, upper bound: 372.2185727
time: 8.87 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -183.6337585, 146.0694580, -183.4802246, 145.8959503, -329.5296631, 329.5495911
1: -153.8881683, 129.3397064, -153.7868500, 129.2041931, -283.0923462, 283.1265259
2: -202.5769958, 131.4073334, -202.4049988, 131.2896576, -333.8666382, 333.8122559
3: -214.6929474, 113.4032822, -214.4713287, 113.3605576, -328.0534973, 327.8745117
4: -197.7884827, 150.9730988, -197.6693420, 150.8564606, -348.6449585, 348.6423950
5: -176.2236938, 136.8193054, -176.1209717, 136.7452240, -312.9689331, 312.9402771
6: -169.2695007, 162.8846588, -169.1090393, 162.7539825, -332.0234680, 331.9937134
7: -184.0311432, 154.3780975, -183.9398651, 154.2812042, -338.3123474, 338.3179626
8: -222.1465149, 152.4044495, -221.9190674, 152.2536163, -374.4001465, 374.3234863
9: -167.6234436, 165.1021271, -167.5107880, 164.9748688, -332.5982361, 332.6129150

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -372.2194083, upper bound: 372.2160333
time: 10.18 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.2146282, upper bound: 372.2146282
time: 7.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.94 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2184162, upper bound: 372.2177288
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2140325, upper bound: 372.2151712
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2161206, upper bound: 372.2124516
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2125272, upper bound: 372.2112188
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2185363, upper bound: 372.2176945
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2128975, upper bound: 372.2149434
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2158053, upper bound: 372.2123259
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2116797, upper bound: 372.2109743
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2219162, upper bound: 372.2213859
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2169333, upper bound: 372.2186283
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2194976, upper bound: 372.2160248
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2155830, upper bound: 372.2147986
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2220912, upper bound: 372.2214364
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2159392, upper bound: 372.2185727
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2194083, upper bound: 372.2160333
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 18.94
Output dim: 2, lower bound: -372.2146282, upper bound: 372.2146282

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -182.9396973, 145.5059052, -180.1600189, 143.2519073, -326.1915894, 325.6659241
1: -153.3903351, 128.9016418, -151.0099030, 126.8750458, -280.2653809, 279.9115601
2: -201.8202362, 130.9585724, -198.7404633, 128.8659821, -330.6861877, 329.6990356
3: -213.9307861, 113.0680695, -210.6281891, 111.3276978, -325.2584839, 323.6962280
4: -196.9981842, 150.4779663, -194.0611725, 148.1023865, -345.1005859, 344.5391235
5: -175.6105804, 136.3684845, -172.9147644, 134.2646637, -309.8752441, 309.2832031
6: -168.6539764, 162.2674713, -166.0400848, 159.8174133, -328.4713745, 328.3074951
7: -183.3957214, 153.8523560, -180.6151886, 151.4949799, -334.8906860, 334.4675293
8: -221.3402557, 151.8979492, -217.9332733, 149.4960327, -370.8363037, 369.8312073
9: -167.0165710, 164.4740753, -164.4786072, 161.9919128, -329.0084839, 328.9526978

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1997992, upper bound: 372.2010553
time: 11.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1984815, upper bound: 372.1968648
time: 9.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -183.3607483, 145.8535461, -174.0762634, 138.4517212, -321.8124695, 319.9297791
1: -153.6663666, 129.1505432, -145.8886261, 122.6017838, -276.2681580, 275.0391846
2: -202.2690125, 131.1682587, -192.0435333, 124.5128555, -326.7818604, 323.2117310
3: -214.3702545, 113.2306519, -203.4680634, 107.5589905, -321.9292603, 316.6987000
4: -197.4695740, 150.7364502, -187.5419464, 143.1023865, -340.5719604, 338.2783813
5: -175.9423676, 136.5972290, -167.0556641, 129.7244110, -305.6667786, 303.6528931
6: -169.0151062, 162.6399231, -160.4431915, 154.4332581, -323.4483337, 323.0830994
7: -183.7307587, 154.1318970, -174.4832001, 146.3626404, -330.0933838, 328.6150818
8: -221.8392792, 152.1696777, -210.6221161, 144.4895020, -366.3287964, 362.7917175
9: -167.3697052, 164.8567657, -158.9367981, 156.5436401, -323.9132996, 323.7935486

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 133

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1971624, upper bound: 372.1959412
time: 8.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1967601, upper bound: 372.1935520
time: 8.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -179.7537994, 142.9823608, -181.3463745, 144.1815491, -323.9353638, 324.3287354
1: -150.6758423, 126.6443100, -151.9312744, 127.6631470, -278.3389893, 278.5755310
2: -198.3048706, 128.6755371, -200.0108490, 129.6533051, -327.9581604, 328.6863403
3: -210.1794739, 111.0866318, -211.9493256, 112.0093994, -322.1888123, 323.0359192
4: -193.6034546, 147.8455811, -195.3622284, 149.0360718, -342.6394348, 343.2077942
5: -172.5561676, 133.9947815, -174.0525055, 135.1136017, -307.6697388, 308.0472412
6: -165.7156525, 159.4450378, -167.1103516, 160.8470306, -326.5626831, 326.5553284
7: -180.2001038, 151.1667175, -181.7647705, 152.4481812, -332.6482849, 332.9314880
8: -217.4787445, 149.2427216, -219.3107452, 150.3903046, -367.8690491, 368.5534668
9: -164.1039276, 161.6146088, -165.5296326, 163.0444489, -327.1483765, 327.1442261

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1992704, upper bound: 372.2007094
time: 10.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1978722, upper bound: 372.1966895
time: 9.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -180.2703247, 143.4019318, -175.3414612, 139.4414368, -319.7117004, 318.7434082
1: -151.0319824, 126.9602890, -146.8767242, 123.4463501, -274.4783325, 273.8370056
2: -198.8570404, 128.9549103, -193.4024963, 125.3582230, -324.2152710, 322.3574219
3: -210.7292023, 111.3080673, -204.8820648, 108.2897186, -319.0189209, 316.1901245
4: -194.1772919, 148.1802063, -188.9295807, 144.0982208, -338.2755127, 337.1098022
5: -172.9781494, 134.2927399, -168.2677917, 130.6322021, -303.6103516, 302.5605164
6: -166.1611328, 159.9006348, -161.5862732, 155.5313110, -321.6924438, 321.4868469
7: -180.6313019, 151.5273743, -175.7139740, 147.3843842, -328.0156860, 327.2413330
8: -218.0868378, 149.5908508, -212.0950317, 145.4448700, -363.5317078, 361.6857605
9: -164.5444031, 162.0833130, -160.0610046, 157.6680908, -322.2124939, 322.1442566

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 133

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1965234, upper bound: 372.1956211
time: 8.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -372.1961479, upper bound: 372.1933837
time: 10.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.02 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1997992, upper bound: 372.2010553
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1984815, upper bound: 372.1968648
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1971624, upper bound: 372.1959412
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1967601, upper bound: 372.1935520
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1992704, upper bound: 372.2007094
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1978722, upper bound: 372.1966895
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1965234, upper bound: 372.1956211
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.02
Output dim: 2, lower bound: -372.1961479, upper bound: 372.1933837
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=375.07427978515625
rel_dist={2: [-372.2697968624731, 372.2697968624731]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2018.90 seconds
