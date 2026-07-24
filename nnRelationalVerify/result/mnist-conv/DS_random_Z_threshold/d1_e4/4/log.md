## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.165768669


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5442555, 0.5442555)
1: (-6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3568206, 0.3568206)
2: (-7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4619796, 0.4619796)
3: (-2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4880099, 0.4880099)
4: (-5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5492563, 0.5492564)
5: (-9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4988909, 0.4988909)
6: (-15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3130085, 0.3130085)
7: (4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2986627, 0.2986629)
8: (-5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4199564, 0.4199564)
9: (-3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3778250, 0.3778253)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.27 + 33.33 = 56.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1674431, upper bound: 0.1674431

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5846
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 5792

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5846

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1674393
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674395, upper bound: 0.1647949
time: 3.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.05 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1674393
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.05
Output dim: 7, lower bound: -0.1674395, upper bound: 0.1647949

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5382074, 0.5373442
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3543544, 0.3540027
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4626019, 0.4629281
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4613597, 0.4572971
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5272555, 0.5301200
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4789243, 0.4759858
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3056911, 0.3046463
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2842801, 0.2860790
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4106355, 0.4090034
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3668664, 0.3682861

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5792

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1647947, upper bound: 0.1670913
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1644470, upper bound: 0.1674392
time: 3.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5373441, 0.5382073
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3540027, 0.3543544
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4629283, 0.4626019
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4572971, 0.4613595
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5301199, 0.5272554
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4759860, 0.4789245
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3046464, 0.3056912
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2860789, 0.2842802
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4090033, 0.4106354
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3682859, 0.3668661

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 5792
type: DSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655441, upper bound: 0.1647948
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674394, upper bound: 0.1628994
time: 3.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.21 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 7, lower bound: -0.1647947, upper bound: 0.1670913
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 7, lower bound: -0.1644470, upper bound: 0.1674392
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 30.21
Output dim: 7, lower bound: -0.1655441, upper bound: 0.1647948
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.21
Output dim: 7, lower bound: -0.1674394, upper bound: 0.1628994

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5002460, 0.4939654
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3483999, 0.3487897
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4430592, 0.4405820
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4344163, 0.4337137
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4973555, 0.4959548
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4515097, 0.4519980
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3045820, 0.3037190
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2826407, 0.2842053
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4022467, 0.4016607
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3430622, 0.3410920

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1628993, upper bound: 0.1670913
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1647946, upper bound: 0.1651959
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4948285, 0.4993827
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3491414, 0.3480482
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4402559, 0.4433856
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4377761, 0.4303539
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4930902, 0.5002201
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4549367, 0.4485712
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3047639, 0.3035371
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2824066, 0.2844394
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4032924, 0.4006147
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3396719, 0.3444823

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1626825, upper bound: 0.1674387
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1644466, upper bound: 0.1656746
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5367047, 0.5376477
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3438976, 0.3455825
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4157224, 0.4213026
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4412265, 0.4429657
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5158253, 0.5147480
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4308414, 0.4271948
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2990474, 0.3007921
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2726369, 0.2688744
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4060988, 0.4072378
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3519416, 0.3481370

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 5792

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655935, upper bound: 0.1628991
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674389, upper bound: 0.1628984
time: 4.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 7, lower bound: -0.1628993, upper bound: 0.1670913
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 7, lower bound: -0.1647946, upper bound: 0.1651959
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 7, lower bound: -0.1626825, upper bound: 0.1674387
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 7, lower bound: -0.1644466, upper bound: 0.1656746
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 7, lower bound: -0.1655935, upper bound: 0.1628991
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 7, lower bound: -0.1674389, upper bound: 0.1628984

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4996864, 0.4933258
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3396287, 0.3386850
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4017596, 0.3933759
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4160221, 0.4176426
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4848471, 0.4816594
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3997800, 0.4068534
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2996830, 0.2981201
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2672346, 0.2707629
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3988495, 0.3987566
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3243333, 0.3247476

Time for backsubstitution: 23.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1628983, upper bound: 0.1670909
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1628990, upper bound: 0.1652455
time: 3.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4948285, 0.4993848
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3491414, 0.3480477
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4402554, 0.4433846
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4377754, 0.4303541
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4930902, 0.5002193
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4549360, 0.4485710
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.3047639, 0.3035378
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2824061, 0.2844396
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4032922, 0.4006145
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3396728, 0.3444824

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 522

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1625506, upper bound: 0.1674385
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1626010, upper bound: 0.1655434
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5367061, 0.5376470
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3438978, 0.3455832
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4157219, 0.4213026
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4412267, 0.4429650
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.5158246, 0.5147480
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4308417, 0.4271946
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2990481, 0.3007921
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2726371, 0.2688740
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.4060998, 0.4072379
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3519415, 0.3481376

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5792

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5792

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674387, upper bound: 0.1625504
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1670910, upper bound: 0.1628981
time: 2.93 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 7, lower bound: -0.1628983, upper bound: 0.1670909
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.15
Output dim: 7, lower bound: -0.1628990, upper bound: 0.1652455
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 7, lower bound: -0.1625506, upper bound: 0.1674385
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.15
Output dim: 7, lower bound: -0.1626010, upper bound: 0.1655434
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 7, lower bound: -0.1674387, upper bound: 0.1625504
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.15
Output dim: 7, lower bound: -0.1670910, upper bound: 0.1628981

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4996864, 0.4933279
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3396292, 0.3386852
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.4017596, 0.3933756
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4160213, 0.4176428
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4848473, 0.4816585
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3997793, 0.4068534
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2996830, 0.2981209
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2672338, 0.2707629
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3988495, 0.3987573
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3243340, 0.3247476

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 773

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1598753, upper bound: 0.1642415
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1600835, upper bound: 0.1639908
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4942690, 0.4987453
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3403707, 0.3379437
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3989563, 0.3961792
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4193811, 0.4142830
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4805818, 0.4859238
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4032063, 0.4034266
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2998649, 0.2979390
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2669997, 0.2709970
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3998957, 0.3977115
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3209437, 0.3281379

Time for backsubstitution: 23.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2516

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1976

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1619354, upper bound: 0.1669126
time: 7.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1620247, upper bound: 0.1668233
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4987453, 0.4942690
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3379436, 0.3403707
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3961792, 0.3989563
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4142830, 0.4193811
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4859238, 0.4805818
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4034266, 0.4032063
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2979391, 0.2998649
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2709970, 0.2669997
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3977118, 0.3998955
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3281378, 0.3209437

Time for backsubstitution: 23.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1788

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1665854, upper bound: 0.1623580
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1672462, upper bound: 0.1616971
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4933277, 0.4996864
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3386850, 0.3396292
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3933756, 0.4017596
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4176428, 0.4160213
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4816585, 0.4848473
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4068537, 0.3997796
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2981209, 0.2996830
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2707629, 0.2672341
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3987575, 0.3988497
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3247474, 0.3243340

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 310

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1262

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1664683, upper bound: 0.1623789
time: 6.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1665781, upper bound: 0.1622550
time: 2.83 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1598753, upper bound: 0.1642415
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1600835, upper bound: 0.1639908
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1619354, upper bound: 0.1669126
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1620247, upper bound: 0.1668233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1665854, upper bound: 0.1623580
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1672462, upper bound: 0.1616971
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1664683, upper bound: 0.1623789
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.17
Output dim: 7, lower bound: -0.1665781, upper bound: 0.1622550

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4834057, 0.4881873
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3337531, 0.3314312
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3881226, 0.3857188
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4124181, 0.4073727
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4694598, 0.4745545
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3997421, 0.3993962
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2997947, 0.2978884
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2609547, 0.2650446
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3934081, 0.3914818
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3160859, 0.3228804

Time for backsubstitution: 23.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1516

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1618719, upper bound: 0.1637097
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1603707, upper bound: 0.1668553
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4837109, 0.4878821
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3338580, 0.3313262
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3884959, 0.3853452
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4124708, 0.4073200
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4692128, 0.4748015
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.3991759, 0.3999624
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2998143, 0.2978689
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2610472, 0.2649521
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3936656, 0.3912240
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3156863, 0.3232802

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1788

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1754

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1615551, upper bound: 0.1660222
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1612240, upper bound: 0.1663536
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4968444, 0.4924269
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3339090, 0.3368210
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3899899, 0.3934200
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4113209, 0.4167314
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4829707, 0.4774189
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4008830, 0.4009454
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2918758, 0.2939913
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2696209, 0.2657382
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3964972, 0.3987173
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3204995, 0.3129784

Time for backsubstitution: 23.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 422

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1650264, upper bound: 0.1574147
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1642080, upper bound: 0.1606865
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4969031, 0.4923681
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3343940, 0.3363360
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3906429, 0.3927670
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.4116330, 0.4164193
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4827609, 0.4776289
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4011657, 0.4006627
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2920653, 0.2938017
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2697353, 0.2656235
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3965335, 0.3986810
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3201724, 0.3133054

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1976

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1432

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1669615, upper bound: 0.1613779
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1669270, upper bound: 0.1614124
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4749150, 0.4831965
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3182306, 0.3203640
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3731508, 0.3807589
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.3956304, 0.3900421
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4823275, 0.4859244
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4050636, 0.3970356
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2963154, 0.2981528
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2660147, 0.2629185
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3722361, 0.3746761
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3186582, 0.3186144

Time for backsubstitution: 23.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 403

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1432

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1661835, upper bound: 0.1620597
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1661491, upper bound: 0.1620943
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.4767883, 0.4812734
1: -6.1904583, -5.3161311, -6.1904583, -5.3161311, -0.3194199, 0.3191091
2: -7.2571950, -6.0329428, -7.2571950, -6.0329428, -0.3723269, 0.3815347
3: -2.2900825, -1.0217202, -2.2900825, -1.0217202, -0.3916636, 0.3939276
4: -5.5683117, -3.8994555, -5.5683117, -3.8994555, -0.4827247, 0.4855164
5: -9.1575279, -7.7887735, -9.1575279, -7.7887735, -0.4041097, 0.3979626
6: -15.1201582, -13.7932243, -15.1201582, -13.7932243, -0.2965844, 0.2978776
7: 4.1257424, 5.1152592, 4.1257424, 5.1152592, -0.2664472, 0.2624732
8: -5.0022507, -3.8621597, -5.0022507, -3.8621597, -0.3745066, 0.3723286
9: -3.3048930, -2.0574970, -3.3048930, -2.0574970, -0.3190278, 0.3182287

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 2516
type: DSZ, layer: 3, pos: 1466
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1153
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 68
type: DSZ, layer: 3, pos: 2811
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2249
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1835
type: DSZ, layer: 3, pos: 431
type: DSZ, layer: 3, pos: 310
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1697
type: DSZ, layer: 3, pos: 1195
type: DSZ, layer: 3, pos: 1432
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1516
type: DSZ, layer: 3, pos: 1754

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 773

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1634700, upper bound: 0.1596836
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1638017, upper bound: 0.1591711
time: 3.25 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 29.45 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1618719, upper bound: 0.1637097
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1603707, upper bound: 0.1668553
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1615551, upper bound: 0.1660222
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1612240, upper bound: 0.1663536
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1650264, upper bound: 0.1574147
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1642080, upper bound: 0.1606865
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1669615, upper bound: 0.1613779
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1669270, upper bound: 0.1614124
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1661835, upper bound: 0.1620597
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1661491, upper bound: 0.1620943
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1634700, upper bound: 0.1596836
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 7, lower bound: -0.1638017, upper bound: 0.1591711

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.59 + 544.94 = 601.53 seconds
