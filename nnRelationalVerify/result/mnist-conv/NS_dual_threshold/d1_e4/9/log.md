## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.177517552


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.5731640, -5.5880413, -6.5731640, -5.5880413, -0.4194586, 0.4194586)
1: (-13.1205788, -12.2291260, -13.1205788, -12.2291260, -0.4893668, 0.4893668)
2: (-8.7633114, -8.0846701, -8.7633114, -8.0846701, -0.4813509, 0.4813509)
3: (-4.1175051, -3.4372332, -4.1175051, -3.4372332, -0.4216642, 0.4216642)
4: (-8.9563923, -8.1901484, -8.9563923, -8.1901484, -0.4117608, 0.4117606)
5: (9.0664110, 9.6858368, 9.0664110, 9.6858368, -0.4606638, 0.4606638)
6: (-11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4267104, 0.4267104)
7: (-8.6385555, -7.8659935, -8.6385555, -7.8659935, -0.3734529, 0.3734527)
8: (-3.6924229, -3.0993862, -3.6924229, -3.0993862, -0.4663706, 0.4663701)
9: (-2.9726958, -2.3287184, -2.9726958, -2.3287184, -0.2750261, 0.2750260)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.57 + 33.63 = 55.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1784096, upper bound: 0.1784103

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 4667
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783459, upper bound: 0.1772847
time: 4.38 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784092, upper bound: 0.1784095
time: 4.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.86 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.86
Output dim: 5, lower bound: -0.1783459, upper bound: 0.1772847
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.86
Output dim: 5, lower bound: -0.1784092, upper bound: 0.1784095

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.5725565, -5.5898871, -6.5730228, -5.5888391, -0.4181545, 0.4175432
1: -13.1066380, -12.2316971, -13.1143541, -12.2292042, -0.4755411, 0.4796238
2: -8.7624226, -8.0850201, -8.7629547, -8.0847092, -0.4804673, 0.4803643
3: -4.1075821, -3.4391983, -4.1130743, -3.4373541, -0.4116673, 0.4153469
4: -8.9556837, -8.1905651, -8.9560976, -8.1902733, -0.4109764, 0.4110827
5: 9.0681744, 9.6783161, 9.0666294, 9.6824694, -0.4557567, 0.4530449
6: -11.0152302, -10.1567116, -11.0204592, -10.1552420, -0.4174390, 0.4205847
7: -8.6316757, -7.8676882, -8.6354742, -7.8661127, -0.3664765, 0.3686686
8: -3.6913376, -3.1021335, -3.6923454, -3.1006134, -0.4630179, 0.4635525
9: -2.9716520, -2.3342781, -2.9726267, -2.3311872, -0.2715392, 0.2694231

Time for backsubstitution: 20.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1772845
time: 4.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1772845
time: 4.28 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.5731654, -5.5880418, -6.5731649, -5.5880413, -0.4194584, 0.4186034
1: -13.1205788, -12.2291241, -13.1205807, -12.2291250, -0.4772398, 0.4891715
2: -8.7633114, -8.0846691, -8.7633133, -8.0846701, -0.4813385, 0.4824886
3: -4.1175046, -3.4372344, -4.1175056, -3.4372334, -0.4122686, 0.4216638
4: -8.9563942, -8.1901493, -8.9563932, -8.1901493, -0.4117587, 0.4119322
5: 9.0664120, 9.6858358, 9.0664101, 9.6858368, -0.4606638, 0.4549270
6: -11.0246582, -10.1552267, -11.0246592, -10.1552277, -0.4186392, 0.4266014
7: -8.6385574, -7.8659940, -8.6385574, -7.8659940, -0.3665180, 0.3734524
8: -3.6924238, -3.0993853, -3.6924233, -3.0993848, -0.4661903, 0.4651346
9: -2.9726963, -2.3287194, -2.9726963, -2.3287191, -0.2750257, 0.2704130

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778236, upper bound: 0.1784093
time: 4.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784089, upper bound: 0.1784092
time: 4.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.57 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.57
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1772845
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.57
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1772845
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.57
Output dim: 5, lower bound: -0.1778236, upper bound: 0.1784093
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.57
Output dim: 5, lower bound: -0.1784089, upper bound: 0.1784092

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.5722237, -5.6008677, -6.5670519, -5.6103048, -0.3961854, 0.4005065
1: -13.1062298, -12.2353706, -13.1118526, -12.2364788, -0.4677567, 0.4733427
2: -8.7621269, -8.0918694, -8.7590361, -8.0983353, -0.4667311, 0.4667826
3: -4.1061211, -3.4393311, -4.1100163, -3.4382730, -0.4082708, 0.4121757
4: -8.9432936, -8.1910019, -8.9315987, -8.1973991, -0.3899808, 0.3859444
5: 9.0688000, 9.6772127, 9.0690289, 9.6803188, -0.4529371, 0.4497004
6: -11.0148115, -10.1615944, -11.0171318, -10.1647511, -0.4071724, 0.4123216
7: -8.6163254, -7.8680358, -8.6056423, -7.8745394, -0.3402667, 0.3383355
8: -3.6911752, -3.1110935, -3.6877680, -3.1180837, -0.4454970, 0.4488835
9: -2.9648795, -2.3344355, -2.9594388, -2.3350043, -0.2606953, 0.2559153

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772844
time: 5.25 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772845
time: 4.30 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.5725560, -5.5898886, -6.5730243, -5.5888443, -0.3961031, 0.4175422
1: -13.1066370, -12.2316971, -13.1143532, -12.2292061, -0.4725955, 0.4796226
2: -8.7624226, -8.0850182, -8.7629547, -8.0847101, -0.4708467, 0.4796429
3: -4.1075826, -3.4391985, -4.1130729, -3.4373550, -0.4113088, 0.4147704
4: -8.9556808, -8.1905661, -8.9560947, -8.1902761, -0.4107361, 0.3868194
5: 9.0681753, 9.6783161, 9.0666294, 9.6824684, -0.4540534, 0.4530430
6: -11.0152302, -10.1567135, -11.0204582, -10.1552429, -0.4081914, 0.4205847
7: -8.6316757, -7.8676877, -8.6354704, -7.8661141, -0.3660107, 0.3386037
8: -3.6913373, -3.1021345, -3.6923447, -3.1006172, -0.4464712, 0.4631834
9: -2.9716523, -2.3342767, -2.9726253, -2.3311887, -0.2715387, 0.2551608

Time for backsubstitution: 21.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1772839, upper bound: 0.1772844
time: 4.33 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1772839, upper bound: 0.1772831
time: 4.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.5728312, -5.5990214, -6.5671916, -5.6095066, -0.3974872, 0.4015656
1: -13.1201668, -12.2327986, -13.1180782, -12.2363977, -0.4694557, 0.4828908
2: -8.7630196, -8.0915184, -8.7593956, -8.0982933, -0.4676037, 0.4689078
3: -4.1160426, -3.4373682, -4.1144471, -3.4381533, -0.4088707, 0.4184911
4: -8.9440069, -8.1905880, -8.9318953, -8.1972752, -0.3907626, 0.3867929
5: 9.0670385, 9.6847305, 9.0688133, 9.6836853, -0.4578438, 0.4515815
6: -11.0242405, -10.1601095, -11.0213318, -10.1647358, -0.4083710, 0.4183366
7: -8.6232052, -7.8663440, -8.6087227, -7.8744230, -0.3402603, 0.3431189
8: -3.6922612, -3.1083469, -3.6878467, -3.1168547, -0.4486694, 0.4504662
9: -2.9659231, -2.3288779, -2.9595051, -2.3325362, -0.2641817, 0.2569050

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4667
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4667

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778236, upper bound: 0.1778240
time: 3.88 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1778236, upper bound: 0.1784092
time: 4.26 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.5731668, -5.5880418, -6.5731649, -5.5880461, -0.3974071, 0.4186022
1: -13.1205788, -12.2291241, -13.1205788, -12.2291279, -0.4742932, 0.4891713
2: -8.7633133, -8.0846701, -8.7633123, -8.0846710, -0.4717169, 0.4817667
3: -4.1175036, -3.4372344, -4.1175051, -3.4372342, -0.4119105, 0.4210846
4: -8.9563932, -8.1901493, -8.9563904, -8.1901493, -0.4115186, 0.3876674
5: 9.0664110, 9.6858368, 9.0664101, 9.6858368, -0.4589620, 0.4549270
6: -11.0246582, -10.1552258, -11.0246582, -10.1552277, -0.4093916, 0.4266012
7: -8.6385555, -7.8659949, -8.6385536, -7.8659954, -0.3660522, 0.3433864
8: -3.6924238, -3.0993862, -3.6924231, -3.0993886, -0.4496417, 0.4647660
9: -2.9726944, -2.3287196, -2.9726942, -2.3287194, -0.2750254, 0.2561513

Time for backsubstitution: 21.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1783459
time: 7.71 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1784097
time: 3.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.81 seconds
NS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772844
NS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772845
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1772839, upper bound: 0.1772844
NS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1772839, upper bound: 0.1772831
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1778236, upper bound: 0.1778240
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1778236, upper bound: 0.1784092
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1783459
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 32.81
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1784097

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5671921, -5.6095061, -6.5671916, -5.6095066, -0.3918498, 0.3909943
1: -13.1180801, -12.2363977, -13.1180782, -12.2363977, -0.4672713, 0.4792030
2: -8.7593956, -8.0982943, -8.7593956, -8.0982933, -0.4631405, 0.4642916
3: -4.1144476, -3.4381523, -4.1144471, -3.4381533, -0.4079180, 0.4173129
4: -8.9318943, -8.1972771, -8.9318953, -8.1972752, -0.3797197, 0.3798933
5: 9.0688124, 9.6836853, 9.0688133, 9.6836853, -0.4561715, 0.4504352
6: -11.0213318, -10.1647358, -11.0213318, -10.1647358, -0.4055431, 0.4135058
7: -8.6087227, -7.8744226, -8.6087227, -7.8744230, -0.3276532, 0.3345881
8: -3.6878462, -3.1168542, -3.6878467, -3.1168547, -0.4438777, 0.4428234
9: -2.9595060, -2.3325365, -2.9595051, -2.3325362, -0.2578398, 0.2532269

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1777611
time: 5.55 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1778245
time: 3.68 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.5731530, -5.5880713, -6.5671916, -5.6095066, -0.3978059, 0.4125843
1: -13.1205635, -12.2291431, -13.1180782, -12.2363977, -0.4697895, 0.4865887
2: -8.7632656, -8.0846853, -8.7593956, -8.0982933, -0.4672937, 0.4733598
3: -4.1174946, -3.4372683, -4.1144471, -3.4381533, -0.4108863, 0.4182827
4: -8.9563904, -8.1901674, -8.9318953, -8.1972752, -0.3914115, 0.3870473
5: 9.0664244, 9.6858206, 9.0688133, 9.6836853, -0.4583015, 0.4527478
6: -11.0246477, -10.1552296, -11.0213318, -10.1647358, -0.4086959, 0.4233789
7: -8.6385517, -7.8659992, -8.6087227, -7.8744230, -0.3402655, 0.3424768
8: -3.6923819, -3.0993974, -3.6878467, -3.1168547, -0.4484878, 0.4531989
9: -2.9726896, -2.3287215, -2.9595051, -2.3325362, -0.2697666, 0.2571591

Time for backsubstitution: 20.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783459
time: 5.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783464
time: 3.85 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.5731668, -5.5880418, -6.5725565, -5.5898914, -0.3955946, 0.4189281
1: -13.1205788, -12.2291241, -13.1066399, -12.2316971, -0.4836528, 0.4754543
2: -8.7633133, -8.0846701, -8.7624197, -8.0850220, -0.4712510, 0.4796019
3: -4.1175036, -3.4372344, -4.1075826, -3.4391987, -0.4194055, 0.4111793
4: -8.9563932, -8.1901493, -8.9556780, -8.1905661, -0.4111791, 0.3867645
5: 9.0664110, 9.6858368, 9.0681763, 9.6783161, -0.4514627, 0.4591105
6: -11.0246582, -10.1552258, -11.0152292, -10.1567144, -0.4158769, 0.4173362
7: -8.6385555, -7.8659949, -8.6316729, -7.8676882, -0.3660231, 0.3365161
8: -3.6924238, -3.0993862, -3.6913378, -3.1021354, -0.4469228, 0.4647498
9: -2.9726944, -2.3287196, -2.9716499, -2.3342776, -0.2694721, 0.2597452

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772607, upper bound: 0.1779877
time: 6.56 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772836, upper bound: 0.1783458
time: 4.17 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -6.5731668, -5.5880418, -6.5731649, -5.5880461, -0.3965521, 0.4186022
1: -13.1205788, -12.2291241, -13.1205788, -12.2291279, -0.4742935, 0.4772387
2: -8.7633133, -8.0846701, -8.7633114, -8.0846710, -0.4728684, 0.4817667
3: -4.1175036, -3.4372344, -4.1175041, -3.4372351, -0.4119096, 0.4116898
4: -8.9563932, -8.1901493, -8.9563904, -8.1901493, -0.4116919, 0.3876674
5: 9.0664110, 9.6858368, 9.0664101, 9.6858358, -0.4532237, 0.4549265
6: -11.0246582, -10.1552258, -11.0246572, -10.1552277, -0.4093914, 0.4186382
7: -8.6385555, -7.8659949, -8.6385517, -7.8659935, -0.3660524, 0.3364515
8: -3.6924238, -3.0993862, -3.6924233, -3.0993893, -0.4485869, 0.4647660
9: -2.9726944, -2.3287196, -2.9726930, -2.3287187, -0.2704127, 0.2561513

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772607, upper bound: 0.1780562
time: 4.23 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772836, upper bound: 0.1784095
time: 4.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.38 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1777611
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1778245
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783459
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783464
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1772607, upper bound: 0.1779877
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1772836, upper bound: 0.1783458
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1772607, upper bound: 0.1780562
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.38
Output dim: 5, lower bound: -0.1772836, upper bound: 0.1784095

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.5671921, -5.6095061, -6.5665836, -5.6113520, -0.3900371, 0.3913240
1: -13.1180801, -12.2363977, -13.1041374, -12.2389698, -0.4766297, 0.4654868
2: -8.7593956, -8.0982943, -8.7585020, -8.0986443, -0.4626741, 0.4621267
3: -4.1144476, -3.4381523, -4.1045246, -3.4401152, -0.4154162, 0.4074080
4: -8.9318943, -8.1972771, -8.9311819, -8.1976891, -0.3793824, 0.3789907
5: 9.0688124, 9.6836853, 9.0705700, 9.6761646, -0.4486728, 0.4546237
6: -11.0213318, -10.1647358, -11.0119028, -10.1662226, -0.4120290, 0.4042435
7: -8.6087227, -7.8744226, -8.6018410, -7.8761153, -0.3329568, 0.3277183
8: -3.6878462, -3.1168542, -3.6867609, -3.1196020, -0.4411588, 0.4428067
9: -2.9595060, -2.3325365, -2.9584622, -2.3380928, -0.2522858, 0.2568215

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1763455, upper bound: 0.1777419
time: 4.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766996, upper bound: 0.1777615
time: 4.57 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.5671921, -5.6095061, -6.5671921, -5.6095061, -0.3909943, 0.3909943
1: -13.1180801, -12.2363977, -13.1180801, -12.2363977, -0.4672713, 0.4672713
2: -8.7593956, -8.0982943, -8.7593956, -8.0982943, -0.4642911, 0.4642911
3: -4.1144476, -3.4381523, -4.1144476, -3.4381523, -0.4079182, 0.4079185
4: -8.9318943, -8.1972771, -8.9318943, -8.1972771, -0.3798935, 0.3798935
5: 9.0688124, 9.6836853, 9.0688124, 9.6836853, -0.4504356, 0.4504352
6: -11.0213318, -10.1647358, -11.0213318, -10.1647358, -0.4055431, 0.4055429
7: -8.6087227, -7.8744226, -8.6087227, -7.8744226, -0.3276536, 0.3276534
8: -3.6878462, -3.1168542, -3.6878462, -3.1168542, -0.4428225, 0.4428225
9: -2.9595060, -2.3325365, -2.9595060, -2.3325365, -0.2532270, 0.2532270

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1763455, upper bound: 0.1778103
time: 4.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766996, upper bound: 0.1778244
time: 4.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.5731530, -5.5880713, -6.5665836, -5.6113520, -0.3959932, 0.4129136
1: -13.1205635, -12.2291431, -13.1041374, -12.2389698, -0.4791477, 0.4728723
2: -8.7632656, -8.0846853, -8.7585020, -8.0986443, -0.4668274, 0.4713516
3: -4.1174946, -3.4372683, -4.1045246, -3.4401152, -0.4183836, 0.4083781
4: -8.9563904, -8.1901674, -8.9311819, -8.1976891, -0.3906021, 0.3861442
5: 9.0664244, 9.6858206, 9.0705700, 9.6761646, -0.4508018, 0.4569373
6: -11.0246477, -10.1552296, -11.0119028, -10.1662226, -0.4151812, 0.4141166
7: -8.6385517, -7.8659992, -8.6018410, -7.8761153, -0.3397458, 0.3355953
8: -3.6923819, -3.0993974, -3.6867609, -3.1196020, -0.4457688, 0.4512582
9: -2.9726896, -2.3287215, -2.9584622, -2.3380928, -0.2642107, 0.2607538

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766795, upper bound: 0.1779879
time: 5.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766995, upper bound: 0.1783458
time: 4.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.5731530, -5.5880713, -6.5671921, -5.6095061, -0.3969507, 0.4125843
1: -13.1205635, -12.2291431, -13.1180801, -12.2363977, -0.4697893, 0.4746566
2: -8.7632656, -8.0846853, -8.7593956, -8.0982943, -0.4684443, 0.4733596
3: -4.1174946, -3.4372683, -4.1144476, -3.4381523, -0.4108860, 0.4088879
4: -8.9563904, -8.1901674, -8.9318943, -8.1972771, -0.3916059, 0.3870475
5: 9.0664244, 9.6858206, 9.0688124, 9.6836853, -0.4525647, 0.4527483
6: -11.0246477, -10.1552296, -11.0213318, -10.1647358, -0.4086957, 0.4154155
7: -8.6385517, -7.8659992, -8.6087227, -7.8744226, -0.3402115, 0.3361955
8: -3.6923819, -3.0993974, -3.6878462, -3.1168542, -0.4474330, 0.4531996
9: -2.9726896, -2.3287215, -2.9595060, -2.3325365, -0.2660660, 0.2571594

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766795, upper bound: 0.1779886
time: 4.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766995, upper bound: 0.1783462
time: 3.88 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5700240, -5.5891271, -6.5708389, -5.5899172, -0.3924923, 0.4160001
1: -13.1177864, -12.2303944, -13.1051283, -12.2317476, -0.4829001, 0.4766560
2: -8.7596436, -8.0862246, -8.7604084, -8.0850744, -0.4675560, 0.4760771
3: -4.1134233, -3.4389076, -4.1053543, -3.4392891, -0.4157908, 0.4073737
4: -8.9559164, -8.1905899, -8.9554176, -8.1906881, -0.4105730, 0.3860378
5: 9.0672569, 9.6849527, 9.0684195, 9.6778326, -0.4503212, 0.4580796
6: -11.0229540, -10.1559095, -11.0143337, -10.1567535, -0.4142313, 0.4158084
7: -8.6336889, -7.8679953, -8.6289177, -7.8677635, -0.3611088, 0.3327966
8: -3.6918252, -3.0997472, -3.6910636, -3.1021926, -0.4463096, 0.4640718
9: -2.9711170, -2.3327460, -2.9715917, -2.3365245, -0.2656416, 0.2556884

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 843

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1769258, upper bound: 0.1779881
time: 4.00 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1769258, upper bound: 0.1779878
time: 7.24 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -6.5731626, -5.5880413, -6.5725536, -5.5898924, -0.3920608, 0.4188530
1: -13.1205769, -12.2291241, -13.1066399, -12.2316961, -0.4826128, 0.4750221
2: -8.7633095, -8.0846710, -8.7624197, -8.0850210, -0.4672704, 0.4796014
3: -4.1175032, -3.4372346, -4.1075807, -3.4391990, -0.4157121, 0.4111788
4: -8.9563913, -8.1901484, -8.9556789, -8.1905661, -0.4111695, 0.3868082
5: 9.0664120, 9.6858358, 9.0681763, 9.6783161, -0.4514627, 0.4590828
6: -11.0246563, -10.1552267, -11.0152292, -10.1567144, -0.4143043, 0.4173360
7: -8.6385517, -7.8659945, -8.6316700, -7.8676896, -0.3602325, 0.3365159
8: -3.6924233, -3.0993865, -3.6913376, -3.1021366, -0.4469175, 0.4647760
9: -2.9726934, -2.3287201, -2.9716511, -2.3342772, -0.2694710, 0.2558300

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 843

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1769258, upper bound: 0.1783223
time: 8.08 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1769258, upper bound: 0.1783457
time: 6.39 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -6.5700240, -5.5891271, -6.5714469, -5.5880718, -0.3934503, 0.4156759
1: -13.1177864, -12.2303944, -13.1190662, -12.2291756, -0.4740050, 0.4784400
2: -8.7596436, -8.0862246, -8.7613001, -8.0847273, -0.4691730, 0.4782424
3: -4.1134233, -3.4389076, -4.1152740, -3.4373236, -0.4082961, 0.4078832
4: -8.9559164, -8.1905899, -8.9561300, -8.1902723, -0.4110873, 0.3869405
5: 9.0672569, 9.6849527, 9.0666523, 9.6853523, -0.4520817, 0.4538975
6: -11.0229540, -10.1559095, -11.0237617, -10.1552668, -0.4077489, 0.4171097
7: -8.6336889, -7.8679953, -8.6357985, -7.8660693, -0.3615758, 0.3327317
8: -3.6918252, -3.0997472, -3.6921511, -3.0994456, -0.4479756, 0.4640870
9: -2.9711170, -2.3327460, -2.9726365, -2.3309667, -0.2665820, 0.2520950

Time for backsubstitution: 22.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 4667

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 843

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1770056, upper bound: 0.1780562
time: 4.15 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1770056, upper bound: 0.1780561
time: 4.32 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -6.5731626, -5.5880413, -6.5731654, -5.5880461, -0.3930182, 0.4185274
1: -13.1205769, -12.2291241, -13.1205778, -12.2291260, -0.4758599, 0.4768057
2: -8.7633095, -8.0846710, -8.7633133, -8.0846701, -0.4688869, 0.4817657
3: -4.1175032, -3.4372346, -4.1175027, -3.4372334, -0.4082174, 0.4116893
4: -8.9563913, -8.1901484, -8.9563894, -8.1901493, -0.4116831, 0.3877115
5: 9.0664120, 9.6858358, 9.0664110, 9.6858368, -0.4532237, 0.4548993
6: -11.0246563, -10.1552267, -11.0246563, -10.1552286, -0.4078188, 0.4186378
7: -8.6385517, -7.8659945, -8.6385527, -7.8659935, -0.3607002, 0.3364515
8: -3.6924233, -3.0993865, -3.6924236, -3.0993884, -0.4485812, 0.4647918
9: -2.9726934, -2.3287201, -2.9726939, -2.3287194, -0.2704117, 0.2522364

Time for backsubstitution: 22.43 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.20 + 560.58 = 615.78 seconds
