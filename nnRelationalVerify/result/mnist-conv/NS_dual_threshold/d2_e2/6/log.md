## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.23908329


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.2076020, -5.5223475, -6.2076020, -5.5223475, -0.3725266, 0.3725266)
1: (-6.1862726, -5.5502210, -6.1862726, -5.5502210, -0.3926246, 0.3926246)
2: (-9.9678802, -9.1544847, -9.9678802, -9.1544847, -0.3700438, 0.3700438)
3: (-8.3231497, -7.4329510, -8.3231497, -7.4329510, -0.5096865, 0.5096865)
4: (-7.5297031, -6.9073687, -7.5297031, -6.9073687, -0.3389231, 0.3389231)
5: (3.6488411, 4.4215889, 3.6488411, 4.4215889, -0.3426521, 0.3426521)
6: (-4.4604273, -3.7868159, -4.4604273, -3.7868159, -0.5508237, 0.5508237)
7: (-9.7019253, -9.0353975, -9.7019253, -9.0353975, -0.4555955, 0.4555957)
8: (0.4475598, 1.0476842, 0.4475598, 1.0476842, -0.4699328, 0.4699328)
9: (-4.8405838, -4.0761609, -4.8405838, -4.0761609, -0.5305023, 0.5305026)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.17 + 33.59 = 56.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2656476, upper bound: 0.2656481

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6130
type: B, layer: 1, pos: 6130
type: B, layer: 1, pos: 456
type: A, layer: 1, pos: 456
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6130

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656466, upper bound: 0.2623201
time: 3.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656464, upper bound: 0.2656465
time: 3.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.69
Output dim: 5, lower bound: -0.2656466, upper bound: 0.2623201
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.69
Output dim: 5, lower bound: -0.2656464, upper bound: 0.2656465

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.2036233, -5.5226059, -6.2069149, -5.5223904, -0.3684978, 0.3716063
1: -6.1830688, -5.5505047, -6.1857166, -5.5502682, -0.3888206, 0.3917401
2: -9.9676285, -9.1546268, -9.9678354, -9.1545086, -0.3692889, 0.3695188
3: -8.3217154, -7.4329619, -8.3228941, -7.4329529, -0.5080795, 0.5093918
4: -7.5293732, -6.9080043, -7.5296450, -6.9074764, -0.3381200, 0.3383775
5: 3.6491151, 4.4175119, 3.6488862, 4.4208827, -0.3417665, 0.3384240
6: -4.4601026, -3.7901821, -4.4603705, -3.7873998, -0.5497754, 0.5470765
7: -9.6988907, -9.0357332, -9.7013979, -9.0354548, -0.4525337, 0.4548967
8: 0.4479880, 1.0475979, 0.4476342, 1.0476711, -0.4686947, 0.4687353
9: -4.8356476, -4.0761676, -4.8397245, -4.0761619, -0.5256176, 0.5296488

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 456
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 6130
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 456

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656439, upper bound: 0.2600267
time: 3.16 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656438, upper bound: 0.2623170
time: 3.11 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.2083025, -5.5126400, -6.2076020, -5.5223489, -0.3722925, 0.3824191
1: -6.1865950, -5.5370426, -6.1862717, -5.5502191, -0.3930409, 0.4024818
2: -9.9678869, -9.1531973, -9.9678793, -9.1544857, -0.3708085, 0.3710231
3: -8.3274269, -7.4286346, -8.3231487, -7.4329510, -0.5153592, 0.5139699
4: -7.5297108, -6.9022536, -7.5297022, -6.9073696, -0.3395574, 0.3443644
5: 3.6368744, 4.4216814, 3.6488404, 4.4215879, -0.3519190, 0.3419735
6: -4.4690990, -3.7850714, -4.4604268, -3.7868176, -0.5593390, 0.5508714
7: -9.7032118, -9.0284300, -9.7019234, -9.0353966, -0.4567668, 0.4626703
8: 0.4450693, 1.0486550, 0.4475617, 1.0476830, -0.4718342, 0.4729919
9: -4.8416948, -4.0663056, -4.8405800, -4.0761609, -0.5307465, 0.5382023

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 456
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 6130
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 456

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656439, upper bound: 0.2632767
time: 3.14 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2656439, upper bound: 0.2656435
time: 3.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.78 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 5, lower bound: -0.2656439, upper bound: 0.2600267
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 5, lower bound: -0.2656438, upper bound: 0.2623170
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 5, lower bound: -0.2656439, upper bound: 0.2632767
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 28.78
Output dim: 5, lower bound: -0.2656439, upper bound: 0.2656435

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2068801, -5.5226097, -0.3680375, 0.3706714
1: -6.1822033, -5.5507789, -6.1855640, -5.5503135, -0.3871880, 0.3911471
2: -9.9672661, -9.1555557, -9.9677782, -9.1546688, -0.3688655, 0.3686192
3: -8.3213196, -7.4351158, -8.3228273, -7.4332981, -0.5072863, 0.5072274
4: -7.5247216, -6.9085312, -7.5288506, -6.9075613, -0.3332616, 0.3373922
5: 3.6502750, 4.4169931, 3.6490724, 4.4207935, -0.3408048, 0.3376659
6: -4.4588127, -3.7906415, -4.4601498, -3.7874773, -0.5480998, 0.5460734
7: -9.6974754, -9.0387363, -9.7011585, -9.0359392, -0.4506576, 0.4531963
8: 0.4504828, 1.0475233, 0.4480553, 1.0476592, -0.4664772, 0.4680595
9: -4.8349495, -4.0790744, -4.8396120, -4.0766287, -0.5245399, 0.5258994

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6130
type: B, layer: 1, pos: 456
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6130

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2600261
time: 3.27 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2600267
time: 3.05 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -6.2053971, -5.5164924, -6.2069120, -5.5223904, -0.3705680, 0.3790896
1: -6.1875591, -5.5399952, -6.1857157, -5.5502682, -0.3956857, 0.4008116
2: -9.9709768, -9.1516180, -9.9678354, -9.1545086, -0.3756632, 0.3723401
3: -8.3361979, -7.4148545, -8.3228922, -7.4329548, -0.5187786, 0.5269408
4: -7.5309644, -6.8963618, -7.5296421, -6.9074779, -0.3386362, 0.3496854
5: 3.6338797, 4.4191647, 3.6488860, 4.4208827, -0.3579562, 0.3404505
6: -4.4668779, -3.7871006, -4.4603705, -3.7874005, -0.5609026, 0.5529873
7: -9.6992893, -9.0209074, -9.7013950, -9.0354557, -0.4547253, 0.4758108
8: 0.4449205, 1.0533481, 0.4476371, 1.0476704, -0.4745905, 0.4817362
9: -4.8472271, -4.0677218, -4.8397245, -4.0761628, -0.5368361, 0.5444689

Time for backsubstitution: 22.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6130
type: B, layer: 1, pos: 456
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 6130

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2623164
time: 3.29 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2623171
time: 3.33 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -6.2080793, -5.5140038, -6.2075682, -5.5225677, -0.3718326, 0.3814855
1: -6.1857276, -5.5373082, -6.1861181, -5.5502663, -0.3914108, 0.4018092
2: -9.9675236, -9.1541281, -9.9678211, -9.1546440, -0.3703854, 0.3701223
3: -8.3270378, -7.4307880, -8.3230839, -7.4332976, -0.5145605, 0.5118039
4: -7.5250559, -6.9027624, -7.5289083, -6.9074545, -0.3346981, 0.3433797
5: 3.6380334, 4.4211612, 3.6490262, 4.4214978, -0.3509214, 0.3412169
6: -4.4678106, -3.7856052, -4.4602070, -3.7868927, -0.5576684, 0.5498700
7: -9.7017956, -9.0314312, -9.7016859, -9.0358820, -0.4548917, 0.4609685
8: 0.4475636, 1.0485809, 0.4479823, 1.0476725, -0.4696150, 0.4723206
9: -4.8409991, -4.0692120, -4.8404665, -4.0766273, -0.5296636, 0.5344260

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 6130
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2632764, upper bound: 0.2632766
time: 3.15 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2632764, upper bound: 0.2632763
time: 3.35 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -6.2100949, -5.5065336, -6.2076025, -5.5223484, -0.3743701, 0.3898927
1: -6.1910906, -5.5265274, -6.1862707, -5.5502205, -0.3998730, 0.4072340
2: -9.9712381, -9.1501961, -9.9678793, -9.1544838, -0.3771828, 0.3738314
3: -8.3419409, -7.4105272, -8.3231468, -7.4329510, -0.5256677, 0.5274248
4: -7.5312996, -6.8907528, -7.5296993, -6.9073687, -0.3400729, 0.3556547
5: 3.6216469, 4.4233470, 3.6488426, 4.4215870, -0.3617521, 0.3439842
6: -4.4758482, -3.7821403, -4.4604254, -3.7868173, -0.5704238, 0.5567842
7: -9.7036171, -9.0136290, -9.7019234, -9.0353985, -0.4589515, 0.4771848
8: 0.4420185, 1.0544055, 0.4475636, 1.0476832, -0.4776623, 0.4832015
9: -4.8533378, -4.0578618, -4.8405790, -4.0761619, -0.5420380, 0.5456176

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6130
type: B, layer: 1, pos: 456
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6130

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2656426
time: 3.18 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2656428
time: 3.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.99 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2600261
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2600267
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2623164
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2623171
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2632764, upper bound: 0.2632766
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2632764, upper bound: 0.2632763
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2656426
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 5, lower bound: -0.2623166, upper bound: 0.2656428

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2035904, -5.5228252, -0.3678465, 0.3673726
1: -6.1822033, -5.5507789, -6.1829157, -5.5505505, -0.3869951, 0.3880346
2: -9.9672661, -9.1555557, -9.9675684, -9.1547871, -0.3685294, 0.3680536
3: -8.3213196, -7.4351158, -8.3216496, -7.4333072, -0.5072715, 0.5059004
4: -7.5247216, -6.9085312, -7.5285783, -6.9080892, -0.3329140, 0.3367875
5: 3.6502750, 4.4169931, 3.6493001, 4.4174209, -0.3373307, 0.3375342
6: -4.4588127, -3.7906415, -4.4598799, -3.7902584, -0.5450592, 0.5457315
7: -9.6974754, -9.0387363, -9.6986513, -9.0362167, -0.4505172, 0.4506946
8: 0.4504828, 1.0475233, 0.4484105, 1.0475864, -0.4656448, 0.4671817
9: -4.8349495, -4.0790744, -4.8355341, -4.0766320, -0.5245352, 0.5218642

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2600261
time: 3.11 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2600267
time: 3.01 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2082601, -5.5128584, -0.3779669, 0.3718227
1: -6.1822033, -5.5507789, -6.1864414, -5.5370951, -0.3969285, 0.3914886
2: -9.9672661, -9.1555557, -9.9678307, -9.1533604, -0.3699151, 0.3683088
3: -8.3213196, -7.4351158, -8.3273640, -7.4289794, -0.5115733, 0.5114527
4: -7.5247216, -6.9085312, -7.5289159, -6.9023499, -0.3387541, 0.3371388
5: 3.6502750, 4.4169931, 3.6370726, 4.4215927, -0.3414783, 0.3469477
6: -4.4588127, -3.7906415, -4.4688473, -3.7851601, -0.5490251, 0.5546298
7: -9.6974754, -9.0387363, -9.7029705, -9.0289154, -0.4577596, 0.4552813
8: 0.4504828, 1.0475233, 0.4454927, 1.0486426, -0.4669626, 0.4701452
9: -4.8349495, -4.0790744, -4.8415809, -4.0667820, -0.5321150, 0.5276918

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600338, upper bound: 0.2600268
time: 3.09 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600338, upper bound: 0.2600268
time: 3.16 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -6.2053971, -5.5164924, -6.2036247, -5.5226073, -0.3703768, 0.3757900
1: -6.1875591, -5.5399952, -6.1830683, -5.5505047, -0.3954930, 0.3976988
2: -9.9709768, -9.1516180, -9.9676275, -9.1546278, -0.3753273, 0.3717746
3: -8.3361979, -7.4148545, -8.3217144, -7.4329634, -0.5187633, 0.5256801
4: -7.5309644, -6.8963618, -7.5293708, -6.9080024, -0.3382884, 0.3490804
5: 3.6338797, 4.4191647, 3.6491151, 4.4175100, -0.3547418, 0.3403190
6: -4.4668779, -3.7871006, -4.4601011, -3.7901819, -0.5578618, 0.5526459
7: -9.6992893, -9.0209074, -9.6988888, -9.0357332, -0.4545846, 0.4734676
8: 0.4449205, 1.0533481, 0.4479909, 1.0475976, -0.4737525, 0.4808912
9: -4.8472271, -4.0677218, -4.8356471, -4.0761681, -0.5368304, 0.5406926

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623164
time: 3.17 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623175
time: 3.09 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -6.2053971, -5.5164924, -6.2082973, -5.5126410, -0.3804970, 0.3802406
1: -6.1875591, -5.5399952, -6.1865950, -5.5370488, -0.4033384, 0.4011519
2: -9.9709768, -9.1516180, -9.9678888, -9.1532011, -0.3767132, 0.3720297
3: -8.3361979, -7.4148545, -8.3274269, -7.4286342, -0.5198171, 0.5311391
4: -7.5309644, -6.8963618, -7.5297065, -6.9022689, -0.3441293, 0.3494319
5: 3.6338797, 4.4191647, 3.6368871, 4.4216814, -0.3579587, 0.3494215
6: -4.4668779, -3.7871006, -4.4690661, -3.7850728, -0.5618281, 0.5607567
7: -9.6992893, -9.0209074, -9.7032089, -9.0284328, -0.4616163, 0.4774342
8: 0.4449205, 1.0533481, 0.4450722, 1.0486562, -0.4750659, 0.4828525
9: -4.8472271, -4.0677218, -4.8416939, -4.0663166, -0.5372066, 0.5455642

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623170
time: 3.19 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623173
time: 3.10 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6.2080793, -5.5140038, -6.2073984, -5.5237107, -0.3710933, 0.3812196
1: -6.1857276, -5.5373082, -6.1854062, -5.5505018, -0.3911309, 0.4004245
2: -9.9675236, -9.1541281, -9.9675159, -9.1554146, -0.3696783, 0.3698906
3: -8.3270378, -7.4307880, -8.3227558, -7.4351044, -0.5128222, 0.5114334
4: -7.5250559, -6.9027624, -7.5250492, -6.9078970, -0.3345554, 0.3393683
5: 3.6380334, 4.4211612, 3.6500027, 4.4210691, -0.3504391, 0.3404974
6: -4.4678106, -3.7856052, -4.4591370, -3.7872748, -0.5570445, 0.5485721
7: -9.7017956, -9.0314312, -9.7005091, -9.0384054, -0.4537325, 0.4596329
8: 0.4475636, 1.0485809, 0.4500551, 1.0476103, -0.4693568, 0.4705181
9: -4.8409991, -4.0692120, -4.8398819, -4.0790677, -0.5266685, 0.5341055

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6130
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6130

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600337, upper bound: 0.2632759
time: 3.13 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2632759
time: 3.63 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6.2080793, -5.5140038, -6.2093797, -5.5162354, -0.3794553, 0.3831154
1: -6.1857276, -5.5373082, -6.1907701, -5.5397148, -0.4005356, 0.4060814
2: -9.9675236, -9.1541281, -9.9712276, -9.1514826, -0.3733530, 0.3765334
3: -8.3270378, -7.4307880, -8.3376408, -7.4148417, -0.5303552, 0.5194378
4: -7.5250559, -6.9027624, -7.5312934, -6.8957467, -0.3460265, 0.3455610
5: 3.6380334, 4.4211612, 3.6336074, 4.4232402, -0.3526471, 0.3575822
6: -4.4678106, -3.7856052, -4.4671888, -3.7837307, -0.5627809, 0.5583577
7: -9.7017956, -9.0314312, -9.7023220, -9.0205679, -0.4754744, 0.4630733
8: 0.4475636, 1.0485809, 0.4445152, 1.0534344, -0.4817057, 0.4779930
9: -4.8409991, -4.0692120, -4.8521791, -4.0677171, -0.5398669, 0.5384047

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6130
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6130

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600337, upper bound: 0.2632765
time: 3.01 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2632760
time: 3.59 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.2100897, -5.5065346, -6.2036247, -5.5226073, -0.3748341, 0.3859005
1: -6.1910915, -5.5265322, -6.1830683, -5.5505047, -0.3989532, 0.4034586
2: -9.9712381, -9.1501970, -9.9676275, -9.1546278, -0.3755823, 0.3731472
3: -8.3419409, -7.4105272, -8.3217144, -7.4329634, -0.5243473, 0.5258212
4: -7.5313005, -6.8907647, -7.5293708, -6.9080024, -0.3386396, 0.3549037
5: 3.6216588, 4.4233465, 3.6491151, 4.4175100, -0.3575385, 0.3444731
6: -4.4758167, -3.7821405, -4.4601011, -3.7901819, -0.5667169, 0.5566149
7: -9.7036161, -9.0136318, -9.6988888, -9.0357332, -0.4591637, 0.4741409
8: 0.4420190, 1.0544050, 0.4479909, 1.0475976, -0.4766469, 0.4813874
9: -4.8533373, -4.0578709, -4.8356471, -4.0761681, -0.5421257, 0.5407321

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656426
time: 2.95 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656437
time: 3.22 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.2100964, -5.5065336, -6.2083015, -5.5126400, -0.3782330, 0.3836298
1: -6.1910915, -5.5265260, -6.1865940, -5.5370417, -0.4062784, 0.4074366
2: -9.9712381, -9.1501961, -9.9678888, -9.1531992, -0.3774520, 0.3739057
3: -8.3419418, -7.4105272, -8.3274279, -7.4286342, -0.5257928, 0.5320175
4: -7.5312996, -6.8907499, -7.5297070, -6.9022503, -0.3441263, 0.3549017
5: 3.6216431, 4.4233475, 3.6368713, 4.4216824, -0.3618163, 0.3482875
6: -4.4758568, -3.7821403, -4.4691067, -3.7850718, -0.5684052, 0.5632336
7: -9.7036180, -9.0136299, -9.7032089, -9.0284290, -0.4627566, 0.4786925
8: 0.4420185, 1.0544055, 0.4450712, 1.0486548, -0.4802845, 0.4845929
9: -4.8533368, -4.0578585, -4.8416948, -4.0663033, -0.5424161, 0.5465505

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 456
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 456

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656426
time: 3.68 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656441
time: 3.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.22 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2600261
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2600267
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600338, upper bound: 0.2600268
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600338, upper bound: 0.2600268
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623164
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623175
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623170
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2623173
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600337, upper bound: 0.2632759
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2632759
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600337, upper bound: 0.2632765
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600336, upper bound: 0.2632760
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656426
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656437
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656426
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 5, lower bound: -0.2600258, upper bound: 0.2656441

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2034206, -5.5239682, -0.3671072, 0.3671073
1: -6.1822033, -5.5507789, -6.1822033, -5.5507789, -0.3867168, 0.3867171
2: -9.9672661, -9.1555557, -9.9672661, -9.1555557, -0.3678216, 0.3678216
3: -8.3213196, -7.4351158, -8.3213196, -7.4351158, -0.5055325, 0.5055323
4: -7.5247216, -6.9085312, -7.5247216, -6.9085312, -0.3327763, 0.3327762
5: 3.6502750, 4.4169931, 3.6502750, 4.4169931, -0.3368146, 0.3368146
6: -4.4588127, -3.7906415, -4.4588127, -3.7906415, -0.5444345, 0.5444345
7: -9.6974754, -9.0387363, -9.6974754, -9.0387363, -0.4493694, 0.4493692
8: 0.4504828, 1.0475233, 0.4504828, 1.0475233, -0.4653866, 0.4653869
9: -4.8349495, -4.0790744, -4.8349495, -4.0790744, -0.5215402, 0.5215402

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 51

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2053971, -5.5164924, -0.3754716, 0.3689799
1: -6.1822033, -5.5507789, -6.1875591, -5.5399952, -0.3961210, 0.3926302
2: -9.9672661, -9.1555557, -9.9709768, -9.1516180, -0.3714979, 0.3744646
3: -8.3213196, -7.4351158, -8.3361979, -7.4148545, -0.5243869, 0.5166726
4: -7.5247216, -6.9085312, -7.5309644, -6.8963618, -0.3442523, 0.3389689
5: 3.6502750, 4.4169931, 3.6338797, 4.4191647, -0.3391277, 0.3541225
6: -4.4588127, -3.7906415, -4.4668779, -3.7871006, -0.5510914, 0.5542274
7: -9.6974754, -9.0387363, -9.6992893, -9.0209074, -0.4711413, 0.4531674
8: 0.4504828, 1.0475233, 0.4449205, 1.0533481, -0.4787364, 0.4728639
9: -4.8349495, -4.0790744, -4.8472271, -4.0677218, -0.5347385, 0.5331483

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 51

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2080755, -5.5140018, -0.3772280, 0.3715576
1: -6.1822033, -5.5507789, -6.1857271, -5.5373135, -0.3966508, 0.3901727
2: -9.9672661, -9.1555557, -9.9675236, -9.1541290, -0.3692062, 0.3680767
3: -8.3213196, -7.4351158, -8.3270378, -7.4307880, -0.5098326, 0.5110824
4: -7.5247216, -6.9085312, -7.5250559, -6.9027753, -0.3386168, 0.3331274
5: 3.6502750, 4.4169931, 3.6380465, 4.4211631, -0.3409619, 0.3462257
6: -4.4588127, -3.7906415, -4.4677806, -3.7856069, -0.5484028, 0.5533361
7: -9.6974754, -9.0387363, -9.7017946, -9.0314350, -0.4566104, 0.4539559
8: 0.4504828, 1.0475233, 0.4475641, 1.0485804, -0.4667046, 0.4683480
9: -4.8349495, -4.0790744, -4.8409972, -4.0692229, -0.5292230, 0.5273643

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 51

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.2034206, -5.5239682, -6.2100897, -5.5065346, -0.3855817, 0.3734373
1: -6.1822033, -5.5507789, -6.1910915, -5.5265322, -0.4017346, 0.3960903
2: -9.9672661, -9.1555557, -9.9712381, -9.1501970, -0.3728707, 0.3747196
3: -8.3213196, -7.4351158, -8.3419409, -7.4105272, -0.5245283, 0.5222564
4: -7.5247216, -6.9085312, -7.5313005, -6.8907647, -0.3500757, 0.3393201
5: 3.6502750, 4.4169931, 3.6216588, 4.4233465, -0.3432837, 0.3569191
6: -4.4588127, -3.7906415, -4.4758167, -3.7821405, -0.5550601, 0.5630822
7: -9.6974754, -9.0387363, -9.7036161, -9.0136318, -0.4718146, 0.4577460
8: 0.4504828, 1.0475233, 0.4420190, 1.0544050, -0.4792328, 0.4757922
9: -4.8349495, -4.0790744, -4.8533373, -4.0578709, -0.5388064, 0.5384092

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 51

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 51

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.2053971, -5.5164924, -6.2034206, -5.5239682, -0.3689799, 0.3754715
1: -6.1875591, -5.5399952, -6.1822033, -5.5507789, -0.3926301, 0.3961213
2: -9.9709768, -9.1516180, -9.9672661, -9.1555557, -0.3744646, 0.3714979
3: -8.3361979, -7.4148545, -8.3213196, -7.4351158, -0.5166726, 0.5243869
4: -7.5309644, -6.8963618, -7.5247216, -6.9085312, -0.3389689, 0.3442525
5: 3.6338797, 4.4191647, 3.6502750, 4.4169931, -0.3541225, 0.3391278
6: -4.4668779, -3.7871006, -4.4588127, -3.7906415, -0.5542274, 0.5510912
7: -9.6992893, -9.0209074, -9.6974754, -9.0387363, -0.4531670, 0.4711413
8: 0.4449205, 1.0533481, 0.4504828, 1.0475233, -0.4728639, 0.4787364
9: -4.8472271, -4.0677218, -4.8349495, -4.0790744, -0.5331483, 0.5347383

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 51

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2578330, upper bound: 0.2623019
time: 2.97 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.77 + 549.64 = 606.41 seconds
