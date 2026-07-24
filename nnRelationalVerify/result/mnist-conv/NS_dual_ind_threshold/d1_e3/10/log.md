## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.20802362000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7637997, 0.7637992)
1: (-14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4929042, 0.4929042)
2: (-8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5185800, 0.5185800)
3: (-8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5946984, 0.5946989)
4: (-1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5654764, 0.5654764)
5: (-11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6780953, 0.6780949)
6: (-13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4749269, 0.4749269)
7: (-3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3887572, 0.3887572)
8: (-5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6434879, 0.6434879)
9: (4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3716483, 0.3716483)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.60 + 35.44 = 58.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2122686, upper bound: 0.2122684

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5872
type: A, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5872

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111463, upper bound: 0.2122092
time: 3.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122679, upper bound: 0.2122683
time: 4.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.39 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.39
Output dim: 9, lower bound: -0.2111463, upper bound: 0.2122092
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.39
Output dim: 9, lower bound: -0.2122679, upper bound: 0.2122683

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.6752672, -12.3876286, -13.6755705, -12.3865585, -0.7582436, 0.7574935
1: -14.4102211, -13.4685087, -14.4133711, -13.4684381, -0.4865723, 0.4897556
2: -8.8882942, -8.1650724, -8.8900967, -8.1649876, -0.5148435, 0.5166140
3: -8.3374329, -7.4191656, -8.3378191, -7.4130845, -0.5880847, 0.5824003
4: -1.9086194, -1.1083684, -1.9122703, -1.1083504, -0.5581813, 0.5618877
5: -11.0210018, -9.9764271, -11.0210495, -9.9680014, -0.6699429, 0.6615152
6: -13.3615160, -12.4486189, -13.3616066, -12.4447203, -0.4696283, 0.4658883
7: -3.2250128, -2.1991768, -3.2295384, -2.1991208, -0.3794103, 0.3838444
8: -5.0614128, -4.1381493, -5.0632677, -4.1379404, -0.6397872, 0.6413651
9: 4.3016567, 5.0196323, 4.2997875, 5.0196619, -0.3679509, 0.3698208

Time for backsubstitution: 20.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5872
type: B, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5872

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111457, upper bound: 0.2111462
time: 3.47 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111457, upper bound: 0.2122092
time: 3.45 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.6793652, -12.3855362, -13.6758556, -12.3855400, -0.7640839, 0.7639165
1: -14.4164410, -13.4619150, -14.4163666, -13.4683714, -0.4895740, 0.4974458
2: -8.8930969, -8.1610041, -8.8918123, -8.1649122, -0.5184889, 0.5224695
3: -8.3514280, -7.4066544, -8.3381882, -7.4073052, -0.6003089, 0.5896244
4: -1.9164511, -1.1007571, -1.9157426, -1.1083347, -0.5629516, 0.5730209
5: -11.0401354, -9.9588699, -11.0210896, -9.9599895, -0.6875172, 0.6700201
6: -13.3704672, -12.4409294, -13.3616886, -12.4410105, -0.4795074, 0.4708595
7: -3.2339258, -2.1895256, -3.2338419, -2.1990676, -0.3836823, 0.3931432
8: -5.0657320, -4.1343241, -5.0650349, -4.1377468, -0.6426425, 0.6469054
9: 4.2978306, 5.0226173, 4.2980037, 5.0196934, -0.3703167, 0.3745942

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5872
type: B, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5872

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122087, upper bound: 0.2111462
time: 3.36 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122087, upper bound: 0.2111457
time: 3.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.87 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.87
Output dim: 9, lower bound: -0.2111457, upper bound: 0.2111462
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.87
Output dim: 9, lower bound: -0.2111457, upper bound: 0.2122092
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.87
Output dim: 9, lower bound: -0.2122087, upper bound: 0.2111462
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.87
Output dim: 9, lower bound: -0.2122087, upper bound: 0.2111457

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13.6752672, -12.3876286, -13.6752672, -12.3876286, -0.7558250, 0.7558250
1: -14.4102211, -13.4685087, -14.4102211, -13.4685087, -0.4865303, 0.4865303
2: -8.8882942, -8.1650724, -8.8882942, -8.1650724, -0.5147457, 0.5147462
3: -8.3374329, -7.4191656, -8.3374329, -7.4191656, -0.5819812, 0.5819812
4: -1.9086194, -1.1083684, -1.9086194, -1.1083684, -0.5581589, 0.5581589
5: -11.0210018, -9.9764271, -11.0210018, -9.9764271, -0.6614652, 0.6614656
6: -13.3615160, -12.4486189, -13.3615160, -12.4486189, -0.4652867, 0.4652867
7: -3.2250128, -2.1991768, -3.2250128, -2.1991768, -0.3791687, 0.3791687
8: -5.0614128, -4.1381493, -5.0614128, -4.1381493, -0.6395726, 0.6395726
9: 4.3016567, 5.0196323, 4.3016567, 5.0196323, -0.3679338, 0.3679338

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2111457
time: 3.27 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111456, upper bound: 0.2111457
time: 3.29 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13.6752672, -12.3876286, -13.6793652, -12.3855362, -0.7579126, 0.7593675
1: -14.4102211, -13.4685087, -14.4164410, -13.4619150, -0.4911516, 0.4928799
2: -8.8882942, -8.1650724, -8.8930969, -8.1610041, -0.5188279, 0.5187097
3: -8.3374329, -7.4191656, -8.3514280, -7.4066544, -0.5922956, 0.5884283
4: -1.9086194, -1.1083684, -1.9164511, -1.1007571, -0.5657296, 0.5657902
5: -11.0210018, -9.9764271, -11.0401354, -9.9588699, -0.6773577, 0.6709743
6: -13.3615160, -12.4486189, -13.3704672, -12.4409294, -0.4729800, 0.4710338
7: -3.2250128, -2.1991768, -3.2339258, -2.1895256, -0.3840069, 0.3875331
8: -5.0614128, -4.1381493, -5.0657320, -4.1343241, -0.6434083, 0.6436858
9: 4.3016567, 5.0196323, 4.2978306, 5.0226173, -0.3709137, 0.3717575

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2122086
time: 3.27 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111456, upper bound: 0.2122086
time: 3.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.6793652, -12.3855362, -13.6752672, -12.3876286, -0.7593670, 0.7579122
1: -14.4164410, -13.4619150, -14.4102211, -13.4685087, -0.4928799, 0.4911516
2: -8.8930969, -8.1610041, -8.8882942, -8.1650724, -0.5187097, 0.5188279
3: -8.3514280, -7.4066544, -8.3374329, -7.4191656, -0.5884285, 0.5922956
4: -1.9164511, -1.1007571, -1.9086194, -1.1083684, -0.5657902, 0.5657296
5: -11.0401354, -9.9588699, -11.0210018, -9.9764271, -0.6709743, 0.6773577
6: -13.3704672, -12.4409294, -13.3615160, -12.4486189, -0.4710340, 0.4729800
7: -3.2339258, -2.1895256, -3.2250128, -2.1991768, -0.3875329, 0.3840066
8: -5.0657320, -4.1343241, -5.0614128, -4.1381493, -0.6436858, 0.6434083
9: 4.2978306, 5.0226173, 4.3016567, 5.0196323, -0.3717575, 0.3709137

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2117490, upper bound: 0.2111457
time: 3.21 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122075, upper bound: 0.2111457
time: 3.42 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.6793652, -12.3855362, -13.6793652, -12.3855362, -0.7642250, 0.7642250
1: -14.4164410, -13.4619150, -14.4164410, -13.4619150, -0.4898415, 0.4898415
2: -8.8930969, -8.1610041, -8.8930969, -8.1610041, -0.5216260, 0.5216260
3: -8.3514280, -7.4066544, -8.3514280, -7.4066544, -0.5926933, 0.5926933
4: -1.9164511, -1.1007571, -1.9164511, -1.1007571, -0.5652323, 0.5652318
5: -11.0401354, -9.9588699, -11.0401354, -9.9588699, -0.6737528, 0.6737533
6: -13.3704672, -12.4409294, -13.3704672, -12.4409294, -0.4711113, 0.4711111
7: -3.2339258, -2.1895256, -3.2339258, -2.1895256, -0.3838873, 0.3838873
8: -5.0657320, -4.1343241, -5.0657320, -4.1343241, -0.6433783, 0.6433783
9: 4.2978306, 5.0226173, 4.2978306, 5.0226173, -0.3705583, 0.3705580

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 567

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2117495, upper bound: 0.2111452
time: 4.69 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122080, upper bound: 0.2111452
time: 3.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.27 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2111457
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2111456, upper bound: 0.2111457
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2122086
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2111456, upper bound: 0.2122086
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2117490, upper bound: 0.2111457
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2122075, upper bound: 0.2111457
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2117495, upper bound: 0.2111452
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 9, lower bound: -0.2122080, upper bound: 0.2111452

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.6724367, -12.3877125, -13.6743736, -12.3876524, -0.7529516, 0.7548337
1: -14.4100962, -13.4694662, -14.4101820, -13.4688091, -0.4859672, 0.4854908
2: -8.8879442, -8.1651049, -8.8881836, -8.1650829, -0.5143747, 0.5145960
3: -8.3372746, -7.4195476, -8.3373852, -7.4192834, -0.5816650, 0.5815344
4: -1.9079763, -1.1083760, -1.9084191, -1.1083703, -0.5574894, 0.5579395
5: -11.0209837, -9.9775314, -11.0209980, -9.9767714, -0.6610980, 0.6603508
6: -13.3604345, -12.4486656, -13.3611765, -12.4486341, -0.4641552, 0.4648180
7: -3.2249398, -2.1992991, -3.2249889, -2.1992149, -0.3789933, 0.3790052
8: -5.0613728, -4.1388702, -5.0613956, -4.1383753, -0.6393285, 0.6388421
9: 4.3025465, 5.0196128, 4.3019361, 5.0196266, -0.3670120, 0.3676348

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2106873
time: 3.28 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2111462
time: 3.24 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.6756029, -12.3808060, -13.6752605, -12.3876286, -0.7551994, 0.7626290
1: -14.4128761, -13.4684486, -14.4102230, -13.4685106, -0.4891605, 0.4862542
2: -8.8884411, -8.1646671, -8.8882904, -8.1650724, -0.5148654, 0.5151544
3: -8.3382053, -7.4189672, -8.3374348, -7.4191647, -0.5827351, 0.5821052
4: -1.9090955, -1.1075060, -1.9086162, -1.1083676, -0.5585260, 0.5590177
5: -11.0239058, -9.9762230, -11.0210018, -9.9764271, -0.6643653, 0.6612554
6: -13.3615923, -12.4457331, -13.3615150, -12.4486198, -0.4649653, 0.4680662
7: -3.2254872, -2.1991282, -3.2250125, -2.1991751, -0.3796265, 0.3791966
8: -5.0627899, -4.1378555, -5.0614100, -4.1381493, -0.6409841, 0.6397109
9: 4.3012037, 5.0211840, 4.3016596, 5.0196319, -0.3680809, 0.3694937

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2106873
time: 3.34 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2111462
time: 3.32 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13.6724367, -12.3877125, -13.6784782, -12.3855667, -0.7550383, 0.7583752
1: -14.4100962, -13.4694662, -14.4164028, -13.4622145, -0.4905250, 0.4918399
2: -8.8879442, -8.1651049, -8.8929853, -8.1610136, -0.5184569, 0.5185595
3: -8.3372746, -7.4195476, -8.3513765, -7.4067731, -0.5919552, 0.5879805
4: -1.9079763, -1.1083760, -1.9162507, -1.1007591, -0.5650635, 0.5655713
5: -11.0209837, -9.9775314, -11.0401297, -9.9592209, -0.6769109, 0.6698561
6: -13.3604345, -12.4486656, -13.3701267, -12.4409466, -0.4718475, 0.4704885
7: -3.2249398, -2.1992991, -3.2339053, -2.1895645, -0.3838254, 0.3873647
8: -5.0613728, -4.1388702, -5.0657206, -4.1345491, -0.6431646, 0.6429553
9: 4.3025465, 5.0196128, 4.2981091, 5.0226111, -0.3699923, 0.3714590

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2117490
time: 3.68 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2122086
time: 3.38 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.6756029, -12.3808060, -13.6793623, -12.3855381, -0.7572870, 0.7661715
1: -14.4128761, -13.4684486, -14.4164391, -13.4619179, -0.4915352, 0.4926038
2: -8.8884411, -8.1646671, -8.8930950, -8.1610031, -0.5189471, 0.5191193
3: -8.3382053, -7.4189672, -8.3514271, -7.4066544, -0.5924435, 0.5885270
4: -1.9090955, -1.1075060, -1.9164509, -1.1007569, -0.5661178, 0.5666490
5: -11.0239058, -9.9762230, -11.0401325, -9.9588747, -0.6773906, 0.6707640
6: -13.3615923, -12.4457331, -13.3704643, -12.4409294, -0.4726586, 0.4712126
7: -3.2254872, -2.1991282, -3.2339277, -2.1895261, -0.3844244, 0.3875600
8: -5.0627899, -4.1378555, -5.0657301, -4.1343250, -0.6448202, 0.6438251
9: 4.3012037, 5.0211840, 4.2978325, 5.0226188, -0.3710608, 0.3731148

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2117497
time: 3.46 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2122086
time: 3.38 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13.6765308, -12.3856277, -13.6743736, -12.3876524, -0.7564936, 0.7569222
1: -14.4163132, -13.4628716, -14.4101820, -13.4688091, -0.4923167, 0.4901049
2: -8.8927460, -8.1610374, -8.8881836, -8.1650829, -0.5183387, 0.5186777
3: -8.3512669, -7.4070368, -8.3373852, -7.4192834, -0.5880885, 0.5918472
4: -1.9158106, -1.1007643, -1.9084191, -1.1083703, -0.5651217, 0.5654764
5: -11.0401154, -9.9599752, -11.0209980, -9.9767714, -0.6705275, 0.6762400
6: -13.3693838, -12.4409809, -13.3611765, -12.4486341, -0.4698930, 0.4725099
7: -3.2338552, -2.1896482, -3.2249889, -2.1992149, -0.3873532, 0.3838377
8: -5.0656948, -4.1350431, -5.0613956, -4.1383753, -0.6434422, 0.6426787
9: 4.2987165, 5.0225983, 4.3019361, 5.0196266, -0.3708365, 0.3706150

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2117490, upper bound: 0.2106873
time: 3.44 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2117490, upper bound: 0.2111462
time: 3.32 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.6797037, -12.3787136, -13.6752605, -12.3876286, -0.7587395, 0.7647161
1: -14.4190979, -13.4618568, -14.4102230, -13.4685106, -0.4939692, 0.4908624
2: -8.8932428, -8.1605978, -8.8882904, -8.1650724, -0.5188293, 0.5192356
3: -8.3521976, -7.4064555, -8.3374348, -7.4191647, -0.5885777, 0.5924141
4: -1.9169307, -1.0998938, -1.9086162, -1.1083676, -0.5661597, 0.5657363
5: -11.0430384, -9.9586687, -11.0210018, -9.9764271, -0.6710067, 0.6771474
6: -13.3705435, -12.4380436, -13.3615150, -12.4486198, -0.4707131, 0.4738777
7: -3.2344036, -2.1894751, -3.2250125, -2.1991751, -0.3879522, 0.3840332
8: -5.0671120, -4.1340322, -5.0614100, -4.1381493, -0.6450996, 0.6435456
9: 4.2973766, 5.0241714, 4.3016596, 5.0196319, -0.3719075, 0.3712032

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122076, upper bound: 0.2106873
time: 3.61 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122076, upper bound: 0.2111462
time: 3.65 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.6765308, -12.3856277, -13.6784782, -12.3855667, -0.7613506, 0.7632341
1: -14.4163132, -13.4628716, -14.4164028, -13.4622145, -0.4892783, 0.4888020
2: -8.8927460, -8.1610374, -8.8929853, -8.1610136, -0.5212569, 0.5214767
3: -8.3512669, -7.4070368, -8.3513765, -7.4067731, -0.5923781, 0.5922470
4: -1.9158106, -1.1007643, -1.9162507, -1.1007591, -0.5645642, 0.5650139
5: -11.0401154, -9.9599752, -11.0401297, -9.9592209, -0.6733851, 0.6726389
6: -13.3693838, -12.4409809, -13.3701267, -12.4409466, -0.4699774, 0.4706416
7: -3.2338552, -2.1896482, -3.2339053, -2.1895645, -0.3837128, 0.3837237
8: -5.0656948, -4.1350431, -5.0657206, -4.1345491, -0.6431351, 0.6426501
9: 4.2987165, 5.0225983, 4.2981091, 5.0226111, -0.3696373, 0.3702600

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2118087, upper bound: 0.2107795
time: 6.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2118087, upper bound: 0.2112390
time: 4.14 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.6797037, -12.3787136, -13.6793623, -12.3855381, -0.7636013, 0.7710242
1: -14.4190979, -13.4618568, -14.4164391, -13.4619179, -0.4924746, 0.4895658
2: -8.8932428, -8.1605978, -8.8930950, -8.1610031, -0.5217466, 0.5220346
3: -8.3521976, -7.4064555, -8.3514271, -7.4066544, -0.5934463, 0.5928164
4: -1.9169307, -1.0998938, -1.9164509, -1.1007569, -0.5656028, 0.5660915
5: -11.0430384, -9.9586687, -11.0401325, -9.9588747, -0.6766524, 0.6735420
6: -13.3705435, -12.4380436, -13.3704643, -12.4409294, -0.4707909, 0.4738870
7: -3.2344036, -2.1894751, -3.2339277, -2.1895261, -0.3843453, 0.3839147
8: -5.0671120, -4.1340322, -5.0657301, -4.1343250, -0.6447911, 0.6435165
9: 4.2973766, 5.0241714, 4.2978325, 5.0226188, -0.3707080, 0.3721175

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 567

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 567

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122673, upper bound: 0.2107795
time: 5.98 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122673, upper bound: 0.2112384
time: 5.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.99 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2106873
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2111462
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2106873
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2111462
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2117490
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2106868, upper bound: 0.2122086
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2117497
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2111458, upper bound: 0.2122086
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2117490, upper bound: 0.2106873
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2117490, upper bound: 0.2111462
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2122076, upper bound: 0.2106873
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2122076, upper bound: 0.2111462
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2118087, upper bound: 0.2107795
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2118087, upper bound: 0.2112390
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2122673, upper bound: 0.2107795
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.99
Output dim: 9, lower bound: -0.2122673, upper bound: 0.2112384

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.6724367, -12.3877125, -13.6724367, -12.3877125, -0.7528801, 0.7528796
1: -14.4100962, -13.4694662, -14.4100962, -13.4694662, -0.4853096, 0.4853096
2: -8.8879442, -8.1651049, -8.8879442, -8.1651049, -0.5143495, 0.5143495
3: -8.3372746, -7.4195476, -8.3372746, -7.4195476, -0.5813994, 0.5813994
4: -1.9079763, -1.1083760, -1.9079763, -1.1083760, -0.5574822, 0.5574822
5: -11.0209837, -9.9775314, -11.0209837, -9.9775314, -0.6603355, 0.6603351
6: -13.3604345, -12.4486656, -13.3604345, -12.4486656, -0.4640675, 0.4640672
7: -3.2249398, -2.1992991, -3.2249398, -2.1992991, -0.3789101, 0.3789101
8: -5.0613728, -4.1388702, -5.0613728, -4.1388702, -0.6388302, 0.6388302
9: 4.3025465, 5.0196128, 4.3025465, 5.0196128, -0.3670037, 0.3670037

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 711
type: A, layer: 3, pos: 954
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2081
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 564
type: A, layer: 3, pos: 2487
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 1443
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1095

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 711

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2043118, upper bound: 0.2077601
time: 3.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2077606, upper bound: 0.2077612
time: 3.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.6724367, -12.3877125, -13.6756029, -12.3808060, -0.7597904, 0.7560735
1: -14.4100962, -13.4694662, -14.4128761, -13.4684486, -0.4863272, 0.4882045
2: -8.8879442, -8.1651049, -8.8884411, -8.1646671, -0.5147967, 0.5148630
3: -8.3372746, -7.4195476, -8.3382053, -7.4189672, -0.5819955, 0.5823498
4: -1.9079763, -1.1083760, -1.9090955, -1.1075060, -0.5583525, 0.5586343
5: -11.0209837, -9.9775314, -11.0239058, -9.9762230, -0.6616249, 0.6632576
6: -13.3604345, -12.4486656, -13.3615923, -12.4457331, -0.4669762, 0.4652326
7: -3.2249398, -2.1992991, -3.2254872, -2.1991282, -0.3790832, 0.3795061
8: -5.0613728, -4.1388702, -5.0627899, -4.1378555, -0.6398482, 0.6402607
9: 4.3025465, 5.0196128, 4.3012037, 5.0211840, -0.3685775, 0.3682821

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 711
type: A, layer: 3, pos: 954
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2081
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 564
type: A, layer: 3, pos: 2487
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 1443
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1095

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 711

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2043118, upper bound: 0.2082188
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2077606, upper bound: 0.2082200
time: 3.47 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13.6756029, -12.3808060, -13.6724367, -12.3877125, -0.7560730, 0.7597914
1: -14.4128761, -13.4684486, -14.4100962, -13.4694662, -0.4882050, 0.4863272
2: -8.8884411, -8.1646671, -8.8879442, -8.1651049, -0.5148630, 0.5147967
3: -8.3382053, -7.4189672, -8.3372746, -7.4195476, -0.5823498, 0.5819955
4: -1.9090955, -1.1075060, -1.9079763, -1.1083760, -0.5586343, 0.5583525
5: -11.0239058, -9.9762230, -11.0209837, -9.9775314, -0.6632576, 0.6616254
6: -13.3615923, -12.4457331, -13.3604345, -12.4486656, -0.4652328, 0.4669759
7: -3.2254872, -2.1991282, -3.2249398, -2.1992991, -0.3795061, 0.3790832
8: -5.0627899, -4.1378555, -5.0613728, -4.1388702, -0.6402607, 0.6398482
9: 4.3012037, 5.0211840, 4.3025465, 5.0196128, -0.3682821, 0.3685775

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 711
type: A, layer: 3, pos: 954
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2081
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 564
type: A, layer: 3, pos: 2487
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 1443
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1095

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 711

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2047706, upper bound: 0.2077599
time: 3.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2082193, upper bound: 0.2077610
time: 3.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13.6756029, -12.3808060, -13.6756029, -12.3808060, -0.7552299, 0.7552299
1: -14.4128761, -13.4684486, -14.4128761, -13.4684486, -0.4867353, 0.4867353
2: -8.8884411, -8.1646671, -8.8884411, -8.1646671, -0.5150356, 0.5150356
3: -8.3382053, -7.4189672, -8.3382053, -7.4189672, -0.5822620, 0.5822620
4: -1.9090955, -1.1075060, -1.9090955, -1.1075060, -0.5585575, 0.5585575
5: -11.0239058, -9.9762230, -11.0239058, -9.9762230, -0.6613879, 0.6613874
6: -13.3615923, -12.4457331, -13.3615923, -12.4457331, -0.4652042, 0.4652042
7: -3.2254872, -2.1991282, -3.2254872, -2.1991282, -0.3796391, 0.3796391
8: -5.0627899, -4.1378555, -5.0627899, -4.1378555, -0.6400380, 0.6400380
9: 4.3012037, 5.0211840, 4.3012037, 5.0211840, -0.3681386, 0.3681383

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 711
type: A, layer: 3, pos: 954
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2081
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 564
type: A, layer: 3, pos: 2487
type: A, layer: 3, pos: 1734
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 1443
type: A, layer: 3, pos: 417
type: A, layer: 3, pos: 1095

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 711

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2047708, upper bound: 0.2077599
time: 4.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2082195, upper bound: 0.2077610
time: 3.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13.6724367, -12.3877125, -13.6765308, -12.3856277, -0.7549667, 0.7564220
1: -14.4100962, -13.4694662, -14.4163132, -13.4628716, -0.4899330, 0.4916587
2: -8.8879442, -8.1651049, -8.8927460, -8.1610374, -0.5184312, 0.5183134
3: -8.3372746, -7.4195476, -8.3512669, -7.4070368, -0.5917146, 0.5878468
4: -1.9079763, -1.1083760, -1.9158106, -1.1007643, -0.5650558, 0.5651145
5: -11.0209837, -9.9775314, -11.0401154, -9.9599752, -0.6762261, 0.6698422
6: -13.3604345, -12.4486656, -13.3693838, -12.4409809, -0.4717598, 0.4698086
7: -3.2249398, -2.1992991, -3.2338552, -2.1896482, -0.3837465, 0.3872740
8: -5.0613728, -4.1388702, -5.0656948, -4.1350431, -0.6426668, 0.6429439
9: 4.3025465, 5.0196128, 4.2987165, 5.0225983, -0.3699841, 0.3708284

Time for backsubstitution: 22.00 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.04 + 558.65 = 616.69 seconds
