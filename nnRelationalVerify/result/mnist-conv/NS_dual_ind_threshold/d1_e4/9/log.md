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
execution time: IAR + RelationalAnalysis = 22.41 + 33.96 = 56.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1784096, upper bound: 0.1784103

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 843

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783459, upper bound: 0.1772847
time: 4.28 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1784092, upper bound: 0.1784095
time: 4.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.82 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.82
Output dim: 5, lower bound: -0.1783459, upper bound: 0.1772847
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.82
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

Time for backsubstitution: 20.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 843

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1772845
time: 4.48 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1772845
time: 4.27 seconds

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

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 843

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1783466
time: 3.92 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1784099
time: 4.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.41
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1772845
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.41
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1772845
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.41
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1783466
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.41
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1784099

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

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4667

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1767004
time: 3.96 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1772845
time: 4.77 seconds

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

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4667

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1767004
time: 4.70 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1772845
time: 3.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.5731654, -5.5880418, -6.5725565, -5.5898871, -0.4176457, 0.4189291
1: -13.1205788, -12.2291241, -13.1066380, -12.2316971, -0.4865983, 0.4754550
2: -8.7633114, -8.0846691, -8.7624226, -8.0850201, -0.4808736, 0.4803233
3: -4.1175046, -3.4372344, -4.1075821, -3.4391983, -0.4197636, 0.4117589
4: -8.9563942, -8.1901493, -8.9556837, -8.1905651, -0.4114201, 0.4110320
5: 9.0664120, 9.6858358, 9.0681744, 9.6783161, -0.4531655, 0.4591107
6: -11.0246582, -10.1552267, -11.0152302, -10.1567116, -0.4251246, 0.4173369
7: -8.6385574, -7.8659940, -8.6316757, -7.8676882, -0.3718207, 0.3665824
8: -3.6924238, -3.0993853, -3.6913376, -3.1021335, -0.4634724, 0.4651184
9: -2.9726963, -2.3287194, -2.9716520, -2.3342781, -0.2694722, 0.2740085

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4667

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1777616
time: 5.64 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1783459
time: 4.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.5731654, -5.5880418, -6.5731654, -5.5880418, -0.4186032, 0.4186032
1: -13.1205788, -12.2291241, -13.1205788, -12.2291241, -0.4772391, 0.4772391
2: -8.7633114, -8.0846691, -8.7633114, -8.0846691, -0.4824882, 0.4824882
3: -4.1175046, -3.4372344, -4.1175046, -3.4372344, -0.4122682, 0.4122684
4: -8.9563942, -8.1901493, -8.9563942, -8.1901493, -0.4119320, 0.4119320
5: 9.0664120, 9.6858358, 9.0664120, 9.6858358, -0.4549274, 0.4549274
6: -11.0246582, -10.1552267, -11.0246582, -10.1552267, -0.4186392, 0.4186392
7: -8.6385574, -7.8659940, -8.6385574, -7.8659940, -0.3665178, 0.3665178
8: -3.6924238, -3.0993853, -3.6924238, -3.0993853, -0.4651341, 0.4651341
9: -2.9726963, -2.3287194, -2.9726963, -2.3287194, -0.2704134, 0.2704133

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4667
type: A, layer: 1, pos: 843

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4667

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1778244
time: 4.22 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1784096
time: 4.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.75 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1767004
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1777613, upper bound: 0.1772845
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1767004
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1783456, upper bound: 0.1772845
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1777616
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1783459
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1772840, upper bound: 0.1778244
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.75
Output dim: 5, lower bound: -0.1772837, upper bound: 0.1784096

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.5665836, -5.6113520, -6.5670519, -5.6103048, -0.3905494, 0.3899360
1: -13.1041374, -12.2389698, -13.1118526, -12.2364788, -0.4655724, 0.4696553
2: -8.7585020, -8.0986443, -8.7590361, -8.0983353, -0.4622693, 0.4621649
3: -4.1045246, -3.4401152, -4.1100163, -3.4382730, -0.4073191, 0.4109988
4: -8.9311819, -8.1976891, -8.9315987, -8.1973991, -0.3789353, 0.3790452
5: 9.0705700, 9.6761646, 9.0690289, 9.6803188, -0.4512696, 0.4485531
6: -11.0119028, -10.1662226, -11.0171318, -10.1647511, -0.4043462, 0.4074905
7: -8.6018410, -7.8761153, -8.6056423, -7.8745394, -0.3276122, 0.3298051
8: -3.6867609, -3.1196020, -3.6877680, -3.1180837, -0.4407063, 0.4412394
9: -2.9584622, -2.3380928, -2.9594388, -2.3350043, -0.2543521, 0.2522365

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 843

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1767004
time: 7.85 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1767005
time: 4.10 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.5725465, -5.5899186, -6.5670519, -5.6103048, -0.3965020, 0.4115260
1: -13.1066227, -12.2317162, -13.1118526, -12.2364788, -0.4680903, 0.4770410
2: -8.7623739, -8.0850334, -8.7590361, -8.0983353, -0.4664216, 0.4715583
3: -4.1075730, -3.4392323, -4.1100163, -3.4382730, -0.4102879, 0.4119661
4: -8.9556780, -8.1905832, -8.9315987, -8.1973991, -0.3906362, 0.3861966
5: 9.0681887, 9.6782990, 9.0690289, 9.6803188, -0.4533939, 0.4508657
6: -11.0152178, -10.1567144, -11.0171318, -10.1647511, -0.4074962, 0.4173636
7: -8.6316738, -7.8676915, -8.6056423, -7.8745394, -0.3402722, 0.3361543
8: -3.6912961, -3.1021442, -3.6877680, -3.1180837, -0.4453158, 0.4519324
9: -2.9716454, -2.3342795, -2.9594388, -2.3350043, -0.2649473, 0.2561687

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772844
time: 5.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772845
time: 4.55 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.5665836, -5.6113520, -6.5730143, -5.5888696, -0.4121389, 0.3958907
1: -13.1041374, -12.2389698, -13.1143360, -12.2292233, -0.4729581, 0.4721730
2: -8.7585020, -8.0986443, -8.7629061, -8.0847235, -0.4716654, 0.4663186
3: -4.1045246, -3.4401152, -4.1130662, -3.4373896, -0.4082866, 0.4139659
4: -8.9311819, -8.1976891, -8.9560938, -8.1902924, -0.3860881, 0.3905873
5: 9.0705700, 9.6761646, 9.0666428, 9.6824532, -0.4535823, 0.4506807
6: -11.0119028, -10.1662226, -11.0204487, -10.1552429, -0.4142187, 0.4106417
7: -8.6018410, -7.8761153, -8.6354713, -7.8661175, -0.3361542, 0.3397442
8: -3.6867609, -3.1196020, -3.6923041, -3.1006260, -0.4512167, 0.4458499
9: -2.9584622, -2.3380928, -2.9726210, -2.3311911, -0.2582844, 0.2650445

Time for backsubstitution: 21.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1766998
time: 6.91 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1767004
time: 4.01 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.5725565, -5.5898914, -6.5730243, -5.5888443, -0.3961034, 0.3954918
1: -13.1066399, -12.2316971, -13.1143532, -12.2292061, -0.4725950, 0.4766786
2: -8.7624197, -8.0850220, -8.7629547, -8.0847101, -0.4708467, 0.4707422
3: -4.1075826, -3.4391987, -4.1130729, -3.4373550, -0.4110885, 0.4147696
4: -8.9556780, -8.1905661, -8.9560947, -8.1902761, -0.3867092, 0.3868194
5: 9.0681763, 9.6783161, 9.0666294, 9.6824684, -0.4540539, 0.4513416
6: -11.0152292, -10.1567144, -11.0204582, -10.1552429, -0.4081910, 0.4113371
7: -8.6316729, -7.8676882, -8.6354704, -7.8661141, -0.3364100, 0.3386035
8: -3.6913378, -3.1021354, -3.6923447, -3.1006172, -0.4464703, 0.4470024
9: -2.9716499, -2.3342776, -2.9726253, -2.3311887, -0.2572758, 0.2551606

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772836
time: 5.69 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772836
time: 6.82 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.5671921, -5.6095061, -6.5722237, -5.6008677, -0.4006076, 0.3969603
1: -13.1180801, -12.2363977, -13.1062298, -12.2353706, -0.4803174, 0.4676709
2: -8.7593956, -8.0982943, -8.7621269, -8.0918694, -0.4672918, 0.4665885
3: -4.1144476, -3.4381523, -4.1061211, -3.4393311, -0.4165926, 0.4083600
4: -8.9318943, -8.1972771, -8.9432936, -8.1910019, -0.3862813, 0.3900361
5: 9.0688124, 9.6836853, 9.0688000, 9.6772127, -0.4498200, 0.4562919
6: -11.0213318, -10.1647358, -11.0148115, -10.1615944, -0.4168601, 0.4070699
7: -8.6087227, -7.8744226, -8.6163254, -7.8680358, -0.3414872, 0.3391789
8: -3.6878462, -3.1168542, -3.6911752, -3.1110935, -0.4488034, 0.4475975
9: -2.9595060, -2.3325365, -2.9648795, -2.3344355, -0.2559648, 0.2631649

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1777614
time: 6.13 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1777611
time: 6.32 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.5731649, -5.5880461, -6.5725560, -5.5898886, -0.4176450, 0.3968782
1: -13.1205788, -12.2291279, -13.1066370, -12.2316971, -0.4865973, 0.4725094
2: -8.7633114, -8.0846710, -8.7624226, -8.0850182, -0.4801512, 0.4707036
3: -4.1175041, -3.4372351, -4.1075826, -3.4391985, -0.4191883, 0.4114008
4: -8.9563904, -8.1901493, -8.9556808, -8.1905661, -0.3871567, 0.4107919
5: 9.0664101, 9.6858358, 9.0681753, 9.6783161, -0.4531641, 0.4574082
6: -11.0246572, -10.1552277, -11.0152302, -10.1567135, -0.4251242, 0.4080889
7: -8.6385517, -7.8659935, -8.6316757, -7.8676877, -0.3417559, 0.3654621
8: -3.6924233, -3.0993893, -3.6913373, -3.1021345, -0.4631033, 0.4485726
9: -2.9726930, -2.3287187, -2.9716523, -2.3342767, -0.2552106, 0.2740079

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 843

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783452
time: 5.57 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783459
time: 6.50 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.5671921, -5.6095061, -6.5728312, -5.5990214, -0.4015656, 0.3966324
1: -13.1180801, -12.2363977, -13.1201668, -12.2327986, -0.4709582, 0.4694557
2: -8.7593956, -8.0982943, -8.7630196, -8.0915184, -0.4689074, 0.4687543
3: -4.1144476, -3.4381523, -4.1160426, -3.4373682, -0.4090955, 0.4088709
4: -8.9318943, -8.1972771, -8.9440069, -8.1905880, -0.3867929, 0.3909361
5: 9.0688124, 9.6836853, 9.0670385, 9.6847305, -0.4515820, 0.4521077
6: -11.0213318, -10.1647358, -11.0242405, -10.1601095, -0.4103739, 0.4083707
7: -8.6087227, -7.8744226, -8.6232052, -7.8663440, -0.3361847, 0.3406355
8: -3.6878462, -3.1168542, -3.6922612, -3.1083469, -0.4504652, 0.4476123
9: -2.9595060, -2.3325365, -2.9659231, -2.3288779, -0.2569051, 0.2595689

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1778244
time: 3.89 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1778240
time: 4.44 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.5731649, -5.5880461, -6.5731668, -5.5880418, -0.4186022, 0.3965523
1: -13.1205788, -12.2291279, -13.1205788, -12.2291241, -0.4772387, 0.4742935
2: -8.7633114, -8.0846710, -8.7633133, -8.0846701, -0.4817667, 0.4728684
3: -4.1175041, -3.4372351, -4.1175036, -3.4372344, -0.4116898, 0.4119098
4: -8.9563904, -8.1901493, -8.9563932, -8.1901493, -0.3876674, 0.4116921
5: 9.0664101, 9.6858358, 9.0664110, 9.6858368, -0.4549265, 0.4532237
6: -11.0246572, -10.1552277, -11.0246582, -10.1552258, -0.4186382, 0.4093916
7: -8.6385517, -7.8659935, -8.6385555, -7.8659949, -0.3364515, 0.3660524
8: -3.6924233, -3.0993893, -3.6924238, -3.0993862, -0.4647655, 0.4485869
9: -2.9726930, -2.3287187, -2.9726944, -2.3287196, -0.2561513, 0.2704127

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4667
type: B, layer: 1, pos: 843

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4667

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1784097
time: 4.03 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1784085
time: 8.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.68 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1767004
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1767005
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772844
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772845
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1766998
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1767004
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772836
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766998, upper bound: 0.1772836
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1777614
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1777611
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783452
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1766997, upper bound: 0.1783459
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1778244
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1778240
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1784097
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 5, lower bound: -0.1767734, upper bound: 0.1784085

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

Time for backsubstitution: 20.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766796, upper bound: 0.1774078
time: 4.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766996, upper bound: 0.1777614
time: 4.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.5671921, -5.6095061, -6.5725465, -5.5899186, -0.4116271, 0.3972766
1: -13.1180801, -12.2363977, -13.1066227, -12.2317162, -0.4807689, 0.4680045
2: -8.7593956, -8.0982943, -8.7623739, -8.0850334, -0.4715796, 0.4662790
3: -4.1144476, -3.4381523, -4.1075730, -3.4392323, -0.4163835, 0.4103768
4: -8.9318943, -8.1972771, -8.9556780, -8.1905832, -0.3865337, 0.3905964
5: 9.0688124, 9.6836853, 9.0681887, 9.6782990, -0.4509854, 0.4567478
6: -11.0213318, -10.1647358, -11.0152178, -10.1567144, -0.4175334, 0.4073937
7: -8.6087227, -7.8744226, -8.6316738, -7.8676915, -0.3361557, 0.3391843
8: -3.6878462, -3.1168542, -3.6912961, -3.1021442, -0.4512987, 0.4474163
9: -2.9595060, -2.3325365, -2.9716454, -2.3342795, -0.2562182, 0.2649783

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1766796, upper bound: 0.1774078
time: 5.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766996, upper bound: 0.1777614
time: 4.07 seconds

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

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 843

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 843

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766795, upper bound: 0.1779878
time: 4.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1766995, upper bound: 0.1783455
time: 4.11 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.36 + 547.49 = 603.85 seconds
