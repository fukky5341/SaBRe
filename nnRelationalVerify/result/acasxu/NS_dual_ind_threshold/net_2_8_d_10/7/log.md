## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 4810.657341545514


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2682.2661133, 2923.5505371, -2682.2661133, 2923.5505371, -5605.8164062, 5605.8164062)
1: (-294.5390930, 204.9020233, -294.5390930, 204.9020233, -499.4411011, 499.4411011)
2: (-466.4013367, 549.2590942, -466.4013367, 549.2590942, -1015.6604004, 1015.6604004)
3: (-542.8253784, 344.3062134, -542.8253784, 344.3062134, -887.1315918, 887.1315918)
4: (-407.1929321, 443.7026672, -407.1929321, 443.7026672, -850.8956299, 850.8956299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.86 + 1.92 = 4.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4810.7054486, upper bound: 4810.7054486

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6842971, upper bound: 4810.6794335
time: 1.34 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.7020311, upper bound: 4810.7020311
time: 0.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 0, lower bound: -4810.6842971, upper bound: 4810.6794335
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 0, lower bound: -4810.7020311, upper bound: 4810.7020311

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2062.9338379, 2321.8532715, -2373.0021973, 2627.1093750, -4690.0429688, 4694.8554688
1: -233.1084290, 156.4414368, -264.2754517, 180.7901764, -413.8984985, 420.7168884
2: -361.0958252, 436.7765198, -414.6475830, 493.8155823, -854.9113770, 851.4240723
3: -425.0379944, 271.7853394, -485.3878784, 308.1733093, -733.2113037, 757.1731567
4: -315.6538086, 352.9238892, -362.5605469, 398.8958435, -714.5496216, 715.4842529

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6650472, upper bound: 4810.6650472
time: 0.60 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6650472, upper bound: 4810.6794335
time: 0.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3436.4514160, 3745.8459473, -2606.5161133, 2844.9096680, -6281.3613281, 6352.3623047
1: -377.4918518, 262.4247131, -286.4108276, 199.1608276, -576.6527100, 548.8353271
2: -598.2296143, 703.7734375, -452.6356201, 534.7014771, -1132.9311523, 1156.4089355
3: -698.0049438, 441.5362854, -526.8779297, 334.6338806, -1032.6387939, 968.4141846
4: -524.1660767, 568.6210938, -395.1089172, 431.7869873, -955.9530029, 963.7299805

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794335, upper bound: 4810.6842971
time: 0.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794335, upper bound: 4810.7020311
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 0, lower bound: -4810.6650472, upper bound: 4810.6650472
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 0, lower bound: -4810.6650472, upper bound: 4810.6794335
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 0, lower bound: -4810.6794335, upper bound: 4810.6842971
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.24
Output dim: 0, lower bound: -4810.6794335, upper bound: 4810.7020311

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2062.9338379, 2321.8532715, -2062.9338379, 2321.8532715, -4384.7871094, 4384.7871094
1: -233.1084290, 156.4414368, -233.1084290, 156.4414368, -389.5498047, 389.5498047
2: -361.0958252, 436.7765198, -361.0958252, 436.7765198, -797.8723145, 797.8723145
3: -425.0379944, 271.7853394, -425.0379944, 271.7853394, -696.8233032, 696.8233032
4: -315.6538086, 352.9238892, -315.6538086, 352.9238892, -668.5775757, 668.5775757

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6139660, upper bound: 4810.6174923
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6643955, upper bound: 4810.6643954
time: 0.65 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2062.9338379, 2321.8532715, -3428.8945312, 3738.8288574, -5801.7626953, 5750.7475586
1: -233.1084290, 156.4414368, -376.7677307, 261.8595276, -494.9679260, 533.2091675
2: -361.0958252, 436.7765198, -596.9448853, 702.4266357, -1063.5224609, 1033.7213135
3: -425.0379944, 271.7853394, -696.6289673, 440.6625977, -865.7005005, 968.4143066
4: -315.6538086, 352.9238892, -523.0415649, 567.5473022, -883.2011108, 875.9653320

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6139660, upper bound: 4810.6386299
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6643955, upper bound: 4810.6791396
time: 0.73 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3434.3752441, 3743.9174805, -2062.9338379, 2321.8532715, -5756.2285156, 5806.8510742
1: -377.2879639, 262.2648926, -233.1084290, 156.4414368, -533.7293701, 495.3732605
2: -597.8886719, 703.3991089, -361.0958252, 436.7765198, -1034.6650391, 1064.4948730
3: -697.6077881, 441.2902527, -425.0379944, 271.7853394, -969.3931274, 866.3282471
4: -523.8634644, 568.3248291, -315.6538086, 352.9238892, -876.7872925, 883.9786377

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6558557, upper bound: 4810.6523459
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6791396, upper bound: 4810.6835040
time: 0.67 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3436.4514160, 3745.8459473, -3436.4514160, 3745.8459473, -7182.2973633, 7182.2973633
1: -377.4918518, 262.4247131, -377.4918518, 262.4247131, -639.9165649, 639.9165039
2: -598.2296143, 703.7734375, -598.2296143, 703.7734375, -1302.0030518, 1302.0030518
3: -698.0049438, 441.5362854, -698.0049438, 441.5362854, -1139.5412598, 1139.5412598
4: -524.1660767, 568.6210938, -524.1660767, 568.6210938, -1092.7871094, 1092.7871094

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6715729, upper bound: 4810.6937816
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6646267, upper bound: 4810.6911489
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.12 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6139660, upper bound: 4810.6174923
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6643955, upper bound: 4810.6643954
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6139660, upper bound: 4810.6386299
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6643955, upper bound: 4810.6791396
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6558557, upper bound: 4810.6523459
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6791396, upper bound: 4810.6835040
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6715729, upper bound: 4810.6937816
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.12
Output dim: 0, lower bound: -4810.6646267, upper bound: 4810.6911489

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2034.2413330, 2291.3896484, -2047.8690186, 2305.7490234, -4339.9897461, 4339.2578125
1: -230.0329590, 154.2146454, -231.4801483, 155.2713623, -385.3043213, 385.6947937
2: -356.1420898, 431.0547791, -358.4889526, 433.7590332, -789.9011230, 789.5437012
3: -419.3778687, 268.1560364, -422.0492249, 269.8668518, -689.2447510, 690.2052612
4: -311.3508911, 348.2998047, -313.3845215, 350.4866333, -661.8372803, 661.6842651

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6504918, upper bound: 4810.6545062
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6438445, upper bound: 4810.6438444
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2034.2413330, 2291.3896484, -3400.3486328, 3709.3217773, -5743.5615234, 5691.7382812
1: -230.0329590, 154.2146454, -373.7400818, 259.6533508, -489.6863098, 527.9547119
2: -356.1420898, 431.0547791, -591.9864502, 696.8728638, -1053.0148926, 1023.0411987
3: -419.3778687, 268.1560364, -690.8681030, 437.1168823, -856.4947510, 959.0241699
4: -311.3508911, 348.2998047, -518.6520386, 563.0931396, -874.4440308, 866.9517822

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6696464, upper bound: 4810.6710943
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6673967, upper bound: 4810.6632656
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3371.4445801, 3678.5085449, -2047.8690186, 2305.7490234, -5677.1923828, 5726.3769531
1: -370.5923462, 257.4086609, -231.4801483, 155.2713623, -525.8637085, 488.8887939
2: -586.9268799, 691.0996094, -358.4889526, 433.7590332, -1020.6859131, 1049.5886230
3: -684.8853149, 433.4365540, -422.0492249, 269.8668518, -954.7521973, 855.4857788
4: -514.1723022, 558.4565430, -313.3845215, 350.4866333, -864.6589355, 871.8410645

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6694775, upper bound: 4810.6774920
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3188.2343750, 3489.7519531, -3421.0183105, 3729.8425293, -6918.0747070, 6910.7690430
1: -351.6493530, 243.0374298, -375.8654785, 261.2221069, -612.8714600, 618.9028931
2: -555.5917358, 655.0114746, -595.5864868, 700.7016602, -1256.2933350, 1250.5977783
3: -649.6787720, 410.6920776, -694.9254761, 439.6126404, -1089.2913818, 1105.6174316
4: -487.3728638, 529.2781372, -521.8579712, 566.1584473, -1053.5312500, 1051.1357422

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6892241
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6965203, upper bound: 4810.6937123
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3360.9421387, 3672.1091309, -3436.4514160, 3745.8459473, -7106.7880859, 7108.5605469
1: -369.9964905, 256.5442200, -377.4918518, 262.4247131, -632.4210815, 634.0360718
2: -585.3003540, 689.7937012, -598.2296143, 703.7734375, -1289.0736084, 1288.0233154
3: -683.6477661, 432.5768127, -698.0049438, 441.5362854, -1125.1839600, 1130.5817871
4: -512.9323120, 557.3615723, -524.1660767, 568.6210938, -1081.5534668, 1081.5275879

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.25 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6504918, upper bound: 4810.6545062
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6438445, upper bound: 4810.6438444
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6696464, upper bound: 4810.6710943
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6673967, upper bound: 4810.6632656
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6694775, upper bound: 4810.6774920
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6892241
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6965203, upper bound: 4810.6937123
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.25
Output dim: 0, lower bound: -4810.6905460, upper bound: 4810.6911489

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2019.2585449, 2276.1411133, -3153.2512207, 3454.2978516, -5473.5561523, 5429.3925781
1: -228.4723816, 153.0658264, -347.9986877, 240.3486328, -468.8210144, 501.0645142
2: -353.5676575, 428.1155701, -549.5336304, 648.3018799, -1001.8695068, 977.6491699
3: -416.3963013, 266.3139954, -642.7217407, 406.3933411, -822.7895508, 909.0357666
4: -309.1145325, 345.9363708, -482.0805054, 523.9110107, -833.0255127, 828.0168457

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6710943
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6700706
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2034.2413330, 2291.3896484, -3323.6738281, 3634.6306152, -5668.8706055, 5615.0629883
1: -230.0329590, 154.2146454, -366.1419067, 253.6722412, -483.7052002, 520.3565674
2: -356.1420898, 431.0547791, -578.8512573, 682.7031860, -1038.8452148, 1009.9058838
3: -419.3778687, 268.1560364, -676.2739868, 428.0431213, -847.4210205, 944.4299316
4: -311.3508911, 348.2998047, -507.3223572, 551.6895752, -863.0402832, 855.6221924

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6673967, upper bound: 4810.6632656
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6673967, upper bound: 4810.6632656
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3356.1232910, 3662.5876465, -1823.2486572, 2076.1093750, -5432.2324219, 5485.8364258
1: -368.9749146, 256.2155151, -208.0687866, 137.9952545, -506.9701538, 464.2842102
2: -584.3011475, 688.0417480, -319.5939636, 389.8293152, -974.1304932, 1007.6356812
3: -681.8229980, 431.5215454, -377.4442139, 242.1144104, -923.9373779, 808.9657593
4: -511.8806458, 556.0053711, -279.7058716, 315.1232605, -827.0038452, 835.7112427

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3371.3972168, 3678.4638672, -1993.8382568, 2251.3674316, -5622.7641602, 5672.3017578
1: -370.5877075, 257.4050293, -225.9472046, 151.0849609, -521.6725464, 483.3522339
2: -586.9190063, 691.0911255, -349.0659180, 423.4160461, -1010.3350830, 1040.1568604
3: -684.8762207, 433.4308167, -411.3136902, 263.3125305, -948.1886597, 844.7445068
4: -514.1652832, 558.4497681, -305.2075806, 342.1621704, -856.3272705, 863.6573486

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3024.1523438, 3329.4956055, -2919.6540527, 3198.9775391, -6223.1289062, 6249.1494141
1: -335.1824341, 230.3755798, -321.5695496, 222.9791107, -558.1615601, 551.9451294
2: -527.0072021, 624.9188232, -507.0437317, 600.5266113, -1127.5338135, 1131.9625244
3: -617.6050415, 391.1602783, -590.6701050, 376.2312927, -993.8363037, 981.8303223
4: -462.7619324, 504.9920044, -443.5668335, 485.2491760, -948.0109253, 948.5588379

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6649631, upper bound: 4810.6654517
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6889508
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3159.4179688, 3460.0026855, -3358.4274902, 3664.7370605, -6824.1542969, 6818.4301758
1: -348.5931091, 240.8122864, -369.2002258, 256.3937683, -604.9868774, 610.0125122
2: -550.5779419, 649.4067383, -584.6820068, 688.4590454, -1239.0368652, 1234.0886230
3: -643.8467407, 407.1083069, -682.2664795, 431.7962341, -1075.6429443, 1089.3747559
4: -482.9349976, 524.7910767, -512.2194824, 556.3352051, -1039.2702637, 1037.0104980

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6926280, upper bound: 4810.6923170
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6959607, upper bound: 4810.6922615
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3360.9421387, 3672.1091309, -3188.2343750, 3489.7519531, -6850.6943359, 6860.3422852
1: -369.9964905, 256.5442200, -351.6493530, 243.0374298, -613.0338135, 608.1934814
2: -585.3003540, 689.7937012, -555.5917358, 655.0114746, -1240.3115234, 1245.3854980
3: -683.6477661, 432.5768127, -649.6787720, 410.6920776, -1094.3397217, 1082.2554932
4: -512.9323120, 557.3615723, -487.3728638, 529.2781372, -1042.2102051, 1044.7343750

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6870131, upper bound: 4810.6774148
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6902570, upper bound: 4810.6907940
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3360.9421387, 3672.1091309, -3360.9421387, 3672.1091309, -7033.0507812, 7033.0502930
1: -369.9964905, 256.5442200, -369.9964905, 256.5442200, -626.5406494, 626.5405884
2: -585.3003540, 689.7937012, -585.3003540, 689.7937012, -1275.0939941, 1275.0939941
3: -683.6477661, 432.5768127, -683.6477661, 432.5768127, -1116.2244873, 1116.2244873
4: -512.9323120, 557.3615723, -512.9323120, 557.3615723, -1070.2939453, 1070.2939453

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6870131, upper bound: 4810.6774148
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6902570, upper bound: 4810.6907940
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.31 seconds
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6710943
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6700706
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6673967, upper bound: 4810.6632656
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6673967, upper bound: 4810.6632656
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6632657, upper bound: 4810.6673967
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6649631, upper bound: 4810.6654517
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6889508
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6926280, upper bound: 4810.6923170
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6959607, upper bound: 4810.6922615
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6870131, upper bound: 4810.6774148
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6902570, upper bound: 4810.6907940
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6870131, upper bound: 4810.6774148
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.31
Output dim: 0, lower bound: -4810.6902570, upper bound: 4810.6907940

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1673.0782471, 1901.3797607, -3064.8374023, 3354.9157715, -5027.9941406, 4966.2167969
1: -190.2744904, 126.3012848, -337.9003296, 233.4053955, -423.6798706, 464.2015991
2: -293.2403564, 355.9310913, -534.1553955, 628.8849487, -922.1251831, 890.0864258
3: -345.5824585, 221.6143799, -624.1804810, 394.6413574, -740.2237549, 845.7948608
4: -256.3844910, 288.4318542, -468.4943542, 508.5811768, -764.9656372, 756.9261475

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6700707
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1986.5432129, 2240.5507812, -3151.7561035, 3452.7746582, -5439.3178711, 5392.3066406
1: -224.8737183, 150.5206604, -347.8427124, 240.2319946, -465.1057129, 498.3633728
2: -347.9067688, 421.2813110, -549.2787476, 648.0101318, -995.9168701, 970.5600586
3: -409.7948608, 262.0865173, -642.4289551, 406.2090454, -816.0039062, 904.5155029
4: -304.1752319, 340.4677734, -481.8588867, 523.6786499, -827.8538818, 822.3265991

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6700707
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1807.5125732, 2059.4375000, -3323.6262207, 3634.5861816, -5442.0986328, 5383.0634766
1: -206.3924255, 136.7733307, -366.1372681, 253.6685944, -460.0609741, 502.9105530
2: -316.8818359, 386.6755981, -578.8434448, 682.6946411, -999.5764160, 965.5190430
3: -374.3474731, 240.1316833, -676.2649536, 428.0375671, -802.3847656, 916.3966064
4: -277.3599548, 312.5778503, -507.3153992, 551.6827393, -829.0427246, 819.8932495

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6667936, upper bound: 4810.6558013
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6662171, upper bound: 4810.6630135
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1980.4969482, 2237.3037109, -3323.6596680, 3634.6171875, -5615.1142578, 5560.9633789
1: -224.5295258, 150.0519562, -366.1405640, 253.6711578, -478.2006836, 516.1925049
2: -346.7618103, 420.7632751, -578.8489380, 682.7006226, -1029.4624023, 999.6121826
3: -408.6852112, 261.6372681, -676.2712402, 428.0414429, -836.7265015, 937.9084473
4: -303.2111511, 340.0190125, -507.3203125, 551.6876221, -854.8986816, 847.3391113

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6667936, upper bound: 4810.6558013
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6662171, upper bound: 4810.6630135
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3124.3981934, 3423.5708008, -1823.2486572, 2076.1093750, -5200.5078125, 5246.8193359
1: -344.8453369, 238.1256409, -208.0687866, 137.9952545, -482.8405762, 446.1944275
2: -544.4772339, 642.5308228, -319.5939636, 389.8293152, -934.3065186, 962.1247559
3: -636.7110596, 402.7054443, -377.4442139, 242.1144104, -878.8254395, 780.1496582
4: -477.5347595, 519.2858887, -279.7058716, 315.1232605, -792.6578979, 798.9917603

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6693911, upper bound: 4810.6765737
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6665385, upper bound: 4810.6756932
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3293.3415527, 3602.5939941, -1823.2486572, 2076.1093750, -5369.4511719, 5425.8427734
1: -362.8563538, 251.3128052, -208.0687866, 137.9952545, -500.8516235, 459.3815308
2: -573.5546875, 676.6915283, -319.5939636, 389.8293152, -963.3840332, 996.2855225
3: -670.0076904, 424.2194824, -377.4442139, 242.1144104, -912.1220703, 801.6636963
4: -502.5445862, 546.8677979, -279.7058716, 315.1232605, -817.6678467, 826.5736694

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6693911, upper bound: 4810.6765737
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6665385, upper bound: 4810.6756932
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3124.4289551, 3423.6003418, -1993.8382568, 2251.3674316, -5375.7963867, 5417.4365234
1: -344.8483582, 238.1280365, -225.9472046, 151.0849609, -495.9332886, 464.0752563
2: -544.4824219, 642.5364380, -349.0659180, 423.4160461, -967.8984375, 991.6023560
3: -636.7167969, 402.7091064, -411.3136902, 263.3125305, -900.0292358, 814.0228271
4: -477.5391846, 519.2903442, -305.2075806, 342.1621704, -819.7011719, 824.4979248

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632063, upper bound: 4810.6666859
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6630135, upper bound: 4810.6662171
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3293.3730469, 3602.6235352, -1993.8382568, 2251.3674316, -5544.7402344, 5596.4609375
1: -362.8595276, 251.3152161, -225.9472046, 151.0849609, -513.9444580, 477.2624207
2: -573.5598755, 676.6971436, -349.0659180, 423.4160461, -996.9759521, 1025.7630615
3: -670.0137329, 424.2232666, -411.3136902, 263.3125305, -933.3261108, 835.5369873
4: -502.5493469, 546.8722534, -305.2075806, 342.1621704, -844.7113647, 852.0798340

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6632063, upper bound: 4810.6666859
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6630135, upper bound: 4810.6662171
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2694.6035156, 2970.8393555, -2832.6865234, 3100.2985840, -5794.9023438, 5803.5239258
1: -298.7012634, 204.4723206, -311.6058350, 216.1341553, -514.8353271, 516.0781250
2: -469.8830261, 555.6296997, -491.8722229, 581.3970337, -1051.2800293, 1047.5018311
3: -550.0623169, 348.5260315, -572.4973755, 364.6636047, -914.7258911, 921.0233765
4: -412.5782166, 450.0379944, -430.2245483, 470.1332397, -882.7114258, 880.2625732

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6649631, upper bound: 4810.6654517
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6649631, upper bound: 4810.6654517
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2990.0158691, 3293.6662598, -2918.6069336, 3197.9023438, -6187.9174805, 6212.2729492
1: -331.5420227, 227.7124481, -321.4588623, 222.8980408, -554.4398193, 549.1713257
2: -521.0836792, 618.0594482, -506.8633423, 600.3162842, -1121.3999023, 1124.9227295
3: -610.8074951, 386.8781738, -590.4597168, 376.1004944, -986.9078979, 977.3378906
4: -457.6644287, 499.5197754, -443.4084167, 485.0822754, -942.7467041, 942.9282227

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6889508
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6889508
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2812.7390137, 3085.4370117, -3265.9575195, 3560.9309082, -6373.6684570, 6351.3940430
1: -310.4373169, 213.6718140, -358.6618347, 249.1223297, -559.5596313, 572.3335571
2: -490.3433533, 577.1901855, -568.5444946, 668.2104492, -1158.5538330, 1145.7346191
3: -572.8878784, 362.5160522, -662.8017578, 419.5165710, -992.4043579, 1025.3177490
4: -430.0547180, 467.4285889, -497.9947205, 540.3580933, -970.4127808, 965.4233398

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6926280, upper bound: 4810.6923170
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6926280, upper bound: 4810.6923170
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3117.2392578, 3416.1166992, -3357.3767090, 3663.6340332, -6780.8730469, 6773.4931641
1: -344.1354980, 237.5126495, -369.0876770, 256.3117981, -600.4472046, 606.6003418
2: -543.3110962, 641.0134277, -584.5019531, 688.2480469, -1231.5590820, 1225.5152588
3: -635.5290527, 401.8543396, -682.0599976, 431.6646423, -1067.1936035, 1083.9141846
4: -476.6808167, 518.0930786, -512.0614624, 556.1666260, -1032.8474121, 1030.1545410

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6959607, upper bound: 4810.6922615
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6959607, upper bound: 4810.6922615
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2861.6113281, 3142.5637207, -3024.1523438, 3329.4956055, -6191.1069336, 6166.7148438
1: -315.8620605, 218.4631958, -335.1824341, 230.3755798, -546.2376709, 553.6455688
2: -497.0489502, 589.8837891, -527.0072021, 624.9188232, -1121.9676514, 1116.8908691
3: -579.7361450, 369.4050903, -617.6050415, 391.1602783, -970.8963623, 987.0101318
4: -435.1672058, 476.6605530, -462.7619324, 504.9920044, -940.1591797, 939.4223633

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6808168, upper bound: 4810.6760739
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6783137
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6838844
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3295.6242676, 3604.7207031, -3159.4179688, 3460.0026855, -6755.6269531, 6764.1367188
1: -363.0807800, 251.4888916, -348.5931091, 240.8122864, -603.8930664, 600.0819092
2: -573.9302979, 677.1045532, -550.5779419, 649.4067383, -1223.3367920, 1227.6822510
3: -670.4439087, 424.4906006, -643.8467407, 407.1083069, -1077.5522461, 1068.3374023
4: -502.8839417, 547.1947021, -482.9349976, 524.7910767, -1027.6750488, 1030.1296387

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6921580, upper bound: 4810.6930092
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6920004, upper bound: 4810.6962755
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2861.6113281, 3142.5637207, -3202.2316895, 3516.0346680, -6377.6455078, 6344.7939453
1: -315.8620605, 218.4631958, -354.0224609, 244.3269958, -560.1890869, 572.4856567
2: -497.0489502, 589.8837891, -557.5688477, 660.5209961, -1157.5699463, 1147.4525146
3: -579.7361450, 369.4050903, -652.6671143, 413.6071167, -993.3432617, 1022.0721436
4: -435.1672058, 476.6605530, -489.1314087, 533.7045898, -968.8718262, 965.7919922

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6794471, upper bound: 4810.6703416
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6761483
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6774148
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3295.6242676, 3604.7207031, -3330.4741211, 3640.9201660, -6936.5444336, 6935.1938477
1: -363.0807800, 251.4888916, -366.7943420, 254.1814728, -617.2622681, 618.2830811
2: -573.9302979, 677.1045532, -579.9992676, 683.9214478, -1257.8515625, 1257.1035156
3: -670.4439087, 424.4906006, -677.4999390, 428.8327942, -1099.2767334, 1101.9904785
4: -502.8839417, 547.1947021, -508.2539673, 552.6591187, -1055.5428467, 1055.4487305

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6888519, upper bound: 4810.6807907
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6899650, upper bound: 4810.6901767
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.43 seconds
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6700707
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6700707
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6667936, upper bound: 4810.6558013
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6662171, upper bound: 4810.6630135
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6667936, upper bound: 4810.6558013
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6662171, upper bound: 4810.6630135
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6693911, upper bound: 4810.6765737
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6665385, upper bound: 4810.6756932
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6693911, upper bound: 4810.6765737
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6665385, upper bound: 4810.6756932
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6632063, upper bound: 4810.6666859
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6630135, upper bound: 4810.6662171
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6632063, upper bound: 4810.6666859
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6630135, upper bound: 4810.6662171
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6649631, upper bound: 4810.6654517
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6649631, upper bound: 4810.6654517
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6889508
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6807103, upper bound: 4810.6889508
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6926280, upper bound: 4810.6923170
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6926280, upper bound: 4810.6923170
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6959607, upper bound: 4810.6922615
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6959607, upper bound: 4810.6922615
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6783137
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6838844
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6921580, upper bound: 4810.6930092
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6920004, upper bound: 4810.6962755
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6761483
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6774148
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6888519, upper bound: 4810.6807907
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -4810.6899650, upper bound: 4810.6901767

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1673.0782471, 1901.3797607, -2806.8105469, 3079.9846191, -4753.0625000, 4708.1904297
1: -190.2744904, 126.3012848, -309.8581848, 213.2218628, -403.4963379, 436.1594849
2: -293.2403564, 355.9310913, -489.3488464, 576.1362915, -869.3765869, 845.2797852
3: -345.5824585, 221.6143799, -571.7308960, 361.8195190, -707.4019775, 793.3452759
4: -256.3844910, 288.4318542, -429.1707153, 466.5885620, -722.9730225, 717.6025391

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6693945
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6693945
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1673.0782471, 1901.3797607, -3109.7851562, 3409.2075195, -5080.8315430, 5011.1650391
1: -190.2744904, 126.3012848, -343.4143982, 236.9536285, -427.2280884, 469.7156982
2: -293.2403564, 355.9310913, -542.0605469, 639.6751099, -932.9153442, 897.9916382
3: -345.5824585, 221.6143799, -634.1629028, 400.9949036, -746.5772705, 855.7772827
4: -256.3844910, 288.4318542, -475.6445923, 517.0271606, -773.4116211, 764.0763550

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6710943
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6710944
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1986.5432129, 2240.5507812, -2807.5544434, 3080.6706543, -5067.2133789, 5048.1049805
1: -224.8737183, 150.5206604, -309.9309082, 213.2782745, -438.1519775, 460.4515686
2: -347.9067688, 421.2813110, -489.4735413, 576.2689819, -924.1757812, 910.7548828
3: -409.7948608, 262.0865173, -571.8760376, 361.9071045, -771.7019653, 833.9625244
4: -304.1752319, 340.4677734, -429.2814941, 466.6941833, -770.8693237, 769.7492676

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1986.5432129, 2240.5507812, -3110.2556152, 3409.6491699, -5396.1918945, 5350.8051758
1: -224.8737183, 150.5206604, -343.4608765, 236.9898682, -461.8634949, 493.9815369
2: -347.9067688, 421.2813110, -542.1373291, 639.7604370, -987.6672363, 963.4186401
3: -409.7948608, 262.0865173, -634.2518921, 401.0511475, -810.8459473, 896.3383789
4: -304.1752319, 340.4677734, -475.7137146, 517.0949707, -821.2701416, 816.1813965

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675452
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675452
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1720.7261963, 1962.9067383, -2939.7326660, 3221.7260742, -4942.4521484, 4902.6391602
1: -196.5609283, 130.0295563, -324.0417480, 223.6467590, -420.2076721, 454.0712891
2: -301.8359070, 367.7881165, -512.0553589, 603.2046509, -905.0405273, 879.8433228
3: -356.4034119, 228.6447906, -597.4536743, 378.8174744, -735.2208862, 826.0984497
4: -264.1651611, 297.6092224, -448.5068665, 488.5088501, -752.6738892, 746.1160889

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6592649
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6592649
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1806.7010498, 2058.5541992, -3287.1840820, 3596.2912598, -5402.9916992, 5345.7382812
1: -206.3023834, 136.7104187, -362.2512512, 250.8299561, -457.1323242, 498.9616394
2: -316.7408447, 386.5045776, -572.5361328, 675.3173218, -992.0580444, 959.0406494
3: -374.1803284, 240.0263519, -669.0365601, 423.4656982, -797.6459961, 909.0629272
4: -277.2361450, 312.4418030, -501.8858948, 545.7967529, -823.0328979, 814.3275146

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6665384
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6665384
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1887.1943359, 2134.6208496, -2939.7453613, 3221.7382812, -5108.9326172, 5074.3662109
1: -214.0391388, 142.8034973, -324.0429993, 223.6477356, -437.6868896, 466.8464661
2: -330.5928345, 400.6843567, -512.0573730, 603.2067871, -933.7996216, 912.7416992
3: -389.3933105, 249.3872833, -597.4561157, 378.8189392, -768.2122803, 846.8433838
4: -289.0062866, 324.1101074, -448.5087280, 488.5106201, -777.5169067, 772.6188354

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6558013
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6558013
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1979.7327881, 2236.4733887, -3287.2231445, 3596.3271484, -5576.0600586, 5523.6962891
1: -224.4452667, 149.9931183, -362.2550354, 250.8329468, -475.2781982, 512.2481689
2: -346.6280823, 420.6025696, -572.5424805, 675.3242188, -1021.9521484, 993.1450195
3: -408.5281982, 261.5384216, -669.0440063, 423.4703674, -831.9984741, 930.5823364
4: -303.0942688, 339.8906555, -501.8915710, 545.8023071, -848.8966064, 841.7822266

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6630135
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6630135
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3038.1733398, 3326.5700684, -1523.1409912, 1748.1695557, -4786.3427734, 4847.4614258
1: -334.9919128, 231.3554077, -174.6670990, 114.7346954, -449.7265930, 406.0224304
2: -529.4856567, 623.5654907, -267.3890686, 326.4685364, -855.9542236, 890.9545288
3: -618.6425781, 391.2408142, -315.9940796, 203.0870209, -821.7295532, 707.2348022
4: -464.3513794, 504.3150330, -234.1064453, 264.7369690, -729.0883789, 738.4213257

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3122.8173828, 3421.9516602, -1792.6195068, 2042.5578613, -5165.3745117, 5214.5703125
1: -344.6795349, 238.0023041, -204.6632843, 135.6240540, -480.3035278, 442.6655273
2: -544.2069092, 642.2210083, -314.2795105, 383.3625183, -927.5693970, 956.5004883
3: -636.3993530, 402.5093689, -371.1805725, 238.1247559, -874.5241089, 773.6898804
4: -477.2966309, 519.0388184, -275.0662231, 309.9628296, -787.2593994, 794.1050415

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3199.0996094, 3496.9758301, -1523.1409912, 1748.1695557, -4947.2690430, 5019.6523438
1: -352.1148987, 243.9060974, -174.6670990, 114.7346954, -466.8496094, 418.5731201
2: -557.1744385, 656.1173706, -267.3890686, 326.4685364, -883.6429443, 923.5064697
3: -650.2645874, 411.7240601, -315.9940796, 203.0870209, -853.3515625, 727.7181396
4: -488.0769958, 530.6217041, -234.1064453, 264.7369690, -752.8138428, 764.7281494

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3292.0441895, 3601.2692871, -1792.6195068, 2042.5578613, -5334.6015625, 5393.8876953
1: -362.7206421, 251.2122345, -204.6632843, 135.6240540, -498.3446045, 455.8754578
2: -573.3331909, 676.4362793, -314.2795105, 383.3625183, -956.6956787, 990.7158203
3: -669.7531738, 424.0586548, -371.1805725, 238.1247559, -907.8779297, 795.2390747
4: -502.3515320, 546.6641846, -275.0662231, 309.9628296, -812.3143311, 821.7304077

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3038.1540527, 3326.5524902, -1654.3500977, 1882.9171143, -4921.0712891, 4980.9023438
1: -334.9900818, 231.3539276, -188.3738251, 124.8278809, -459.8179016, 419.7277527
2: -529.4824219, 623.5620117, -289.9538879, 352.3502808, -881.8327026, 913.5158081
3: -618.6388550, 391.2385254, -341.8434143, 219.3643341, -838.0031738, 733.0819092
4: -464.3486328, 504.3121948, -253.5507202, 285.5903320, -749.9389038, 757.8629150

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3122.8496094, 3421.9824219, -1965.9659424, 2220.4897461, -5343.3393555, 5387.9482422
1: -344.6827087, 238.0048523, -222.8190765, 148.9404297, -493.6231079, 460.8238831
2: -544.2123413, 642.2269287, -344.2121277, 417.4189148, -961.6311646, 986.4390259
3: -636.4057007, 402.5133057, -405.5833740, 259.6482544, -896.0538940, 808.0966797
4: -477.3013306, 519.0437012, -300.9592896, 337.3838196, -814.6851807, 820.0028687

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3199.0798340, 3496.9582520, -1654.3500977, 1882.9171143, -5081.9970703, 5151.3081055
1: -352.1130676, 243.9046021, -188.3738251, 124.8278809, -476.9409180, 432.2783813
2: -557.1711426, 656.1138916, -289.9538879, 352.3502808, -909.5214233, 946.0676880
3: -650.2607422, 411.7218323, -341.8434143, 219.3643341, -869.6250610, 753.5652466
4: -488.0741272, 530.6188965, -253.5507202, 285.5903320, -773.6644287, 784.1696167

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3292.0773926, 3601.3007812, -1965.9659424, 2220.4897461, -5512.5673828, 5567.2661133
1: -362.7239990, 251.2148132, -222.8190765, 148.9404297, -511.6644287, 474.0338440
2: -573.3386841, 676.4423828, -344.2121277, 417.4189148, -990.7575684, 1020.6543579
3: -669.7596436, 424.0626221, -405.5833740, 259.6482544, -929.4077759, 829.6459961
4: -502.3565369, 546.6689453, -300.9592896, 337.3838196, -839.7403564, 847.6282349

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2694.6035156, 2970.8393555, -2610.2011719, 2873.9133301, -5568.5166016, 5581.0395508
1: -298.7012634, 204.4723206, -288.7283630, 198.7114105, -497.4126587, 493.2006836
2: -469.8830261, 555.6296997, -453.6834412, 538.2161865, -1008.0991211, 1009.3130493
3: -550.0623169, 348.5260315, -529.2684937, 337.3226624, -887.3848877, 877.7945557
4: -412.5782166, 450.0379944, -397.4736633, 435.3359985, -847.9141846, 847.5116577

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2694.6035156, 2970.8393555, -2771.4113770, 3041.1042480, -5735.7060547, 5742.2509766
1: -298.7012634, 204.4723206, -305.5876770, 211.3526001, -510.0538635, 510.0599976
2: -469.8830261, 555.6296997, -481.3675232, 570.1921997, -1040.0751953, 1036.9971924
3: -550.0623169, 348.5260315, -560.9054565, 357.4490967, -907.5114136, 909.4313354
4: -412.5782166, 450.0379944, -421.3347778, 461.0986328, -873.6768799, 871.3726196

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2990.0158691, 3293.6662598, -2697.6599121, 2974.1601562, -5964.1752930, 5991.3261719
1: -331.5420227, 227.7124481, -298.8069458, 205.5823059, -537.1242065, 526.5194092
2: -521.0836792, 618.0594482, -468.9187622, 557.6242065, -1078.7076416, 1086.9781494
3: -610.8074951, 386.8781738, -547.5292358, 349.0224915, -959.8298950, 934.4074097
4: -457.6644287, 499.5197754, -410.8734131, 450.6953125, -908.3597412, 910.3931885

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6684598, upper bound: 4810.6807884
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6773989
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6889508
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2990.0158691, 3293.6662598, -2860.7106934, 3141.6171875, -6131.6328125, 6154.3769531
1: -331.5420227, 227.7124481, -315.7659302, 218.3935242, -549.9353638, 543.4783936
2: -521.0836792, 618.0594482, -496.8933105, 589.7012329, -1110.7849121, 1114.9527588
3: -610.8074951, 386.8781738, -579.5588989, 369.2920227, -980.0994873, 966.4370728
4: -457.6644287, 499.5197754, -435.0338745, 476.5148010, -934.1791992, 934.5536499

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6684598, upper bound: 4810.6807983
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6773989
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6889508
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2812.7390137, 3085.4370117, -3041.5764160, 3329.7597656, -6141.7407227, 6127.0136719
1: -310.4373169, 213.6718140, -335.3277283, 231.6171265, -542.0544434, 548.9992676
2: -490.3433533, 577.1901855, -530.0413208, 624.1812134, -1114.5245361, 1107.2314453
3: -572.8878784, 362.5160522, -619.2859497, 391.6464539, -964.5343018, 981.8020020
4: -430.0547180, 467.4285889, -464.8397827, 504.8046875, -934.8593750, 932.2683716

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787620, upper bound: 4810.6622261
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6627665, upper bound: 4810.6608147
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2812.7390137, 3085.4370117, -3202.5544434, 3500.1816406, -6312.9184570, 6287.9912109
1: -310.4373169, 213.6718140, -352.4544373, 244.1717529, -554.6090698, 566.1262207
2: -490.3433533, 577.1901855, -557.7406006, 656.7407227, -1147.0841064, 1134.9307861
3: -572.8878784, 362.5160522, -650.9221802, 412.1330261, -985.0208740, 1013.4382324
4: -430.0547180, 467.4285889, -488.5782776, 531.1155396, -961.1702881, 956.0068359

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6787620, upper bound: 4810.6743897
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6627664, upper bound: 4810.6741761
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3117.2392578, 3416.1166992, -3125.3654785, 3424.3498535, -6541.5883789, 6541.4814453
1: -344.1354980, 237.5126495, -344.9318237, 238.1985168, -582.3339233, 582.4443970
2: -543.3110962, 641.0134277, -544.6237793, 642.6837769, -1185.9948730, 1185.6372070
3: -635.5290527, 401.8543396, -636.8820801, 402.8140869, -1038.3431396, 1038.7364502
4: -476.6808167, 518.0930786, -477.6629028, 519.4069824, -996.0877686, 995.7559204

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6785544
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6922615
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3117.2392578, 3416.1166992, -3294.6384277, 3603.6862793, -6720.9243164, 6710.7548828
1: -344.1354980, 237.5126495, -362.9757996, 251.4123840, -595.5478516, 600.4884644
2: -543.3110962, 641.0134277, -573.7602539, 676.9060059, -1220.2169189, 1214.7735596
3: -635.5290527, 401.8543396, -670.2492065, 424.3667908, -1059.8957520, 1072.1035156
4: -476.6808167, 518.0930786, -502.7373657, 547.0360107, -1023.7167969, 1020.8304443

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6785544
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6922615
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2861.6113281, 3142.5637207, -2698.7390137, 2975.2585449, -5836.8696289, 5841.3012695
1: -315.8620605, 218.4631958, -298.9197083, 205.6672211, -521.5292969, 517.3829346
2: -497.0489502, 589.8837891, -469.1051025, 557.8375854, -1054.8863525, 1058.9886475
3: -579.7361450, 369.4050903, -547.7449341, 349.1563416, -928.8924561, 917.1500244
4: -435.1672058, 476.6605530, -411.0358582, 450.8647156, -886.0319214, 887.6964111

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766911, upper bound: 4810.6769478
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6783137
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2861.6113281, 3142.5637207, -3120.5100098, 3419.9753418, -6281.5869141, 6263.0727539
1: -315.8620605, 218.4631958, -344.4769897, 237.8340912, -553.6961670, 562.9401855
2: -497.0489502, 589.8837891, -543.7925415, 641.8481445, -1138.8970947, 1133.6760254
3: -579.7361450, 369.4050903, -636.0225830, 402.2664795, -982.0026245, 1005.4276733
4: -435.1672058, 476.6605530, -476.9877319, 518.7375488, -953.9047241, 953.6483154

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6766911, upper bound: 4810.6833294
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6838844
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3202.5544434, 3500.1816406, -2812.7390137, 3085.4370117, -6287.9912109, 6312.9184570
1: -352.4544373, 244.1717529, -310.4373169, 213.6718140, -566.1262207, 554.6090698
2: -557.7406006, 656.7407227, -490.3433533, 577.1901855, -1134.9307861, 1147.0841064
3: -650.9221802, 412.1330261, -572.8878784, 362.5160522, -1013.4382324, 985.0208740
4: -488.5782776, 531.1155396, -430.0547180, 467.4285889, -956.0068359, 961.1702881

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6930070
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6930092
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3294.6384277, 3603.6862793, -3117.2392578, 3416.1166992, -6710.7548828, 6720.9238281
1: -362.9757996, 251.4123840, -344.1354980, 237.5126495, -600.4884644, 595.5478516
2: -573.7602539, 676.9060059, -543.3110962, 641.0134277, -1214.7735596, 1220.2169189
3: -670.2492065, 424.3667908, -635.5290527, 401.8543396, -1072.1033936, 1059.8957520
4: -502.7373657, 547.0360107, -476.6808167, 518.0930786, -1020.8304443, 1023.7167969

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6961280
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6962755
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2861.6113281, 3142.5637207, -2861.6113281, 3142.5637207, -6004.1748047, 6004.1748047
1: -315.8620605, 218.4631958, -315.8620605, 218.4631958, -534.3252563, 534.3252563
2: -497.0489502, 589.8837891, -497.0489502, 589.8837891, -1086.9324951, 1086.9324951
3: -579.7361450, 369.4050903, -579.7361450, 369.4050903, -949.1412354, 949.1412354
4: -435.1672058, 476.6605530, -435.1672058, 476.6605530, -911.8277588, 911.8277588

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756881, upper bound: 4810.6749625
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6761483
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2861.6113281, 3142.5637207, -3291.2397461, 3600.6740723, -6462.2851562, 6433.8027344
1: -315.8620605, 218.4631958, -362.6654358, 251.1651764, -567.0272217, 581.1286011
2: -497.0489502, 589.8837891, -573.1787109, 676.3206787, -1173.3695068, 1163.0621338
3: -579.7361450, 369.4050903, -669.6723022, 423.9881897, -1003.7243652, 1039.0772705
4: -435.1672058, 476.6605530, -502.3095703, 546.5701904, -981.7374268, 978.9700317

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756881, upper bound: 4810.6773160
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6774148
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3202.5544434, 3500.1816406, -2943.9392090, 3225.6557617, -6428.2099609, 6444.1206055
1: -352.4544373, 244.1717529, -324.4566956, 223.9700623, -576.4244995, 568.6284180
2: -557.7406006, 656.7407227, -512.7515869, 603.9658203, -1161.7064209, 1169.4921875
3: -650.9221802, 412.1330261, -598.2608032, 379.3175354, -1030.2397461, 1010.3937988
4: -488.5782776, 531.1155396, -449.1331787, 489.1124573, -977.6906128, 980.2486572

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6807907
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6807907
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3294.6384277, 3603.6862793, -3294.7612305, 3603.3906250, -6898.0288086, 6898.4462891
1: -362.9757996, 251.4123840, -362.9884338, 251.4039917, -614.3797607, 614.4008179
2: -573.7602539, 676.9060059, -573.8256226, 676.6909790, -1250.4511719, 1250.7312012
3: -670.2492065, 424.3667908, -670.4315186, 424.3470154, -1094.5961914, 1094.7980957
4: -502.7373657, 547.0360107, -502.9453125, 546.8864136, -1049.6237793, 1049.9813232

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6888543
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6901767
time: 0.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.71 seconds
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6693945
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6693945
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6710943
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6680099, upper bound: 4810.6710944
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675080
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675452
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6678545, upper bound: 4810.6675452
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6592649
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6592649
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6665384
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6665384
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6558013
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6558013
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6630135
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6662172, upper bound: 4810.6630135
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6709240, upper bound: 4810.6776443
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6592649, upper bound: 4810.6756932
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6675080, upper bound: 4810.6678545
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6558013, upper bound: 4810.6662171
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6773989
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6889508
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6773989
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6889508
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6787620, upper bound: 4810.6622261
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6627665, upper bound: 4810.6608147
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6787620, upper bound: 4810.6743897
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6627664, upper bound: 4810.6741761
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6785544
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6922615
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6785544
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6781924, upper bound: 4810.6922615
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6766911, upper bound: 4810.6769478
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6783137
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6766911, upper bound: 4810.6833294
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6768472, upper bound: 4810.6838844
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6930070
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6930092
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6961280
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6812745, upper bound: 4810.6962755
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756881, upper bound: 4810.6749625
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6761483
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756881, upper bound: 4810.6773160
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6756999, upper bound: 4810.6774148
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6807907
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6807907
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6888543
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.71
Output dim: 0, lower bound: -4810.6797290, upper bound: 4810.6901767

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1507.9995117, 1732.1308594, -2806.7182617, 3079.8999023, -4587.8984375, 4538.8491211
1: -173.0534973, 113.5718613, -309.8491821, 213.2148438, -386.2683411, 423.4210205
2: -264.7515869, 323.4611816, -489.3332825, 576.1199951, -840.8714600, 812.7944336
3: -312.9757996, 201.1850433, -571.7127686, 361.8086243, -674.7844238, 772.8978271
4: -231.8137817, 262.2998962, -429.1568909, 466.5754395, -698.3892212, 691.4567871

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6652366, upper bound: 4810.6693425
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6647290, upper bound: 4810.6523045
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1641.2375488, 1869.1540527, -2806.6872559, 3079.8703613, -4721.1079102, 4675.8413086
1: -186.9884949, 123.8148651, -309.8461609, 213.2124939, -400.2008972, 433.6610107
2: -287.6911011, 349.7573853, -489.3280945, 576.1143799, -863.8054810, 839.0854492
3: -339.2583313, 217.7254944, -571.7069092, 361.8049316, -701.0632324, 789.4323120
4: -251.5863647, 283.4991150, -429.1522827, 466.5710144, -718.1573486, 712.6513062

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6652366, upper bound: 4810.6693425
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6647290, upper bound: 4810.6523045
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1507.9995117, 1732.1308594, -3109.7265625, 3409.1523438, -4913.1630859, 4841.8574219
1: -173.0534973, 113.5718613, -343.4085999, 236.9491272, -410.0026245, 456.9804382
2: -264.7515869, 323.4611816, -542.0510254, 639.6644287, -904.1918335, 865.5122070
3: -312.9757996, 201.1850433, -634.1519165, 400.9879761, -713.9637451, 835.3369751
4: -231.8137817, 262.2998962, -475.6359863, 517.0187378, -748.7148438, 737.9359131

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6558514, upper bound: 4810.6563750
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6674260, upper bound: 4810.6705877
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1641.2375488, 1869.1540527, -3109.7067871, 3409.1337891, -5049.2675781, 4978.8608398
1: -186.9884949, 123.8148651, -343.4066772, 236.9476013, -423.9360657, 467.2215271
2: -287.6911011, 349.7573853, -542.0478516, 639.6608276, -927.3519287, 891.8052368
3: -339.2583313, 217.7254944, -634.1481323, 400.9855957, -740.2438965, 851.8735962
4: -251.5863647, 283.4991150, -475.6330566, 517.0159302, -768.6022339, 759.1320190

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6558514, upper bound: 4810.6563750
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6674260, upper bound: 4810.6705877
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1776.5800781, 2025.4161377, -2807.4135742, 3080.5405273, -4857.1201172, 4832.8295898
1: -202.9456635, 134.3760223, -309.9170532, 213.2675934, -416.2132568, 444.2930908
2: -311.5124207, 380.1314087, -489.4497986, 576.2437134, -887.7561035, 869.5811768
3: -368.0287781, 236.0931702, -571.8485107, 361.8904419, -729.9191895, 807.9416504
4: -272.6796265, 307.3490906, -429.2604370, 466.6741333, -739.3536987, 736.6094971

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6660812, upper bound: 4810.6594465
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6681442, upper bound: 4810.6667966
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1952.2457275, 2206.1159668, -2807.4724121, 3080.5954590, -5032.8398438, 5013.5883789
1: -221.3683777, 147.8832855, -309.9228821, 213.2720642, -434.6403809, 457.8061523
2: -341.8371277, 414.7083130, -489.4597168, 576.2542725, -918.0914307, 904.1680298
3: -402.8758850, 257.9342957, -571.8600464, 361.8973999, -764.7733154, 829.7943115
4: -298.9022522, 335.1932983, -429.2692566, 466.6824646, -765.5845947, 764.4625244

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6660812, upper bound: 4810.6594465
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6681442, upper bound: 4810.6667966
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1776.5800781, 2025.4161377, -3110.1660156, 3409.5659180, -5186.1445312, 5135.5815430
1: -202.9456635, 134.3760223, -343.4521179, 236.9829407, -439.9285889, 477.8281250
2: -311.5124207, 380.1314087, -542.1228027, 639.7440796, -951.2564697, 922.2542114
3: -368.0287781, 236.0931702, -634.2350464, 401.0404358, -769.0692139, 870.3282471
4: -272.6796265, 307.3490906, -475.7006226, 517.0820923, -789.7617188, 783.0496826

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6206414, upper bound: 4810.6286499
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6206414, upper bound: 4810.6675452
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1952.2457275, 2206.1159668, -3110.2041016, 3409.6000977, -5361.8452148, 5316.3198242
1: -221.3683777, 147.8832855, -343.4557495, 236.9858551, -458.3541565, 491.3390198
2: -341.8371277, 414.7083130, -542.1290283, 639.7509155, -981.5880127, 956.8372192
3: -402.8758850, 257.9342957, -634.2422485, 401.0449219, -803.9207764, 892.1765137
4: -298.9022522, 335.1932983, -475.7061462, 517.0874634, -815.9895630, 810.8992920

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6206414, upper bound: 4810.6291593
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6206415, upper bound: 4810.6675452
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1507.9995117, 1732.1308594, -2938.9255371, 3220.9726562, -4728.9707031, 4671.0566406
1: -173.0534973, 113.5718613, -323.9620972, 223.5847321, -396.6382446, 437.5339355
2: -264.7515869, 323.4611816, -511.9217834, 603.0584717, -867.8099976, 835.3829346
3: -312.9757996, 201.1850433, -597.2988281, 378.7215881, -691.6973877, 798.4838867
4: -231.8137817, 262.2998962, -448.3866882, 488.3930054, -720.2066650, 710.6865845

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6763801, upper bound: 4810.6592649
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757720, upper bound: 4810.6592649
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1776.5800781, 2025.4161377, -2939.5256348, 3221.5334473, -4998.1127930, 4964.9418945
1: -202.9456635, 134.3760223, -324.0213623, 223.6308594, -426.5765381, 458.3973999
2: -311.5124207, 380.1314087, -512.0210571, 603.1671753, -914.6795654, 892.1524658
3: -368.0287781, 236.0931702, -597.4140625, 378.7929077, -746.8215942, 833.5072021
4: -272.6796265, 307.3490906, -448.4761353, 488.4790955, -761.1586914, 755.8251953

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6763801, upper bound: 4810.6592649
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6757720, upper bound: 4810.6592649
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1507.9995117, 1732.1308594, -3286.6972656, 3595.8366699, -5101.5859375, 5018.8281250
1: -173.0534973, 113.5718613, -362.2033997, 250.7924957, -423.8460083, 475.7752380
2: -264.7515869, 323.4611816, -572.4564209, 675.2290649, -939.9805908, 895.9176025
3: -312.9757996, 201.1850433, -668.9443970, 423.4077148, -736.3834839, 870.1294556
4: -231.8137817, 262.2998962, -501.8140869, 545.7268066, -777.5405884, 764.1140137

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6662500
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6745153, upper bound: 4810.6634782
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1776.5800781, 2025.4161377, -3287.1540527, 3596.2634277, -5372.8417969, 5312.5703125
1: -202.9456635, 134.3760223, -362.2483521, 250.8276672, -453.7733154, 496.6243896
2: -311.5124207, 380.1314087, -572.5313721, 675.3117676, -986.8242188, 952.6627197
3: -368.0287781, 236.0931702, -669.0310669, 423.4622192, -791.4909058, 905.1242676
4: -272.6796265, 307.3490906, -501.8815918, 545.7924194, -818.4720459, 809.2305908

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6756932, upper bound: 4810.6593190
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4810.6745153, upper bound: 4810.6593190
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1641.2375488, 1869.1540527, -2938.8984375, 3220.9470215, -4862.1845703, 4808.0527344
1: -186.9884949, 123.8148651, -323.9594421, 223.5826721, -410.5711365, 447.7742615
2: -287.6911011, 349.7573853, -511.9172363, 603.0535278, -890.7446289, 861.6746216
3: -339.2583313, 217.7254944, -597.2937622, 378.7183838, -717.9766235, 815.0191650
4: -251.5863647, 283.4991150, -448.3828430, 488.3891602, -739.9753418, 731.8817749

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6491476, upper bound: 4810.6468278
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6491470, upper bound: 4810.6408013
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1952.0197754, 2205.9060059, -2939.5771484, 3221.5810547, -5173.6005859, 5145.4833984
1: -221.3466644, 147.8666534, -324.0263977, 223.6347656, -444.9814148, 471.8929749
2: -341.7968750, 414.6697998, -512.0294800, 603.1762695, -944.9731445, 926.6992798
3: -402.8301697, 257.9087830, -597.4237061, 378.7989502, -781.6291504, 855.3323364
4: -298.8671265, 335.1614380, -448.4836731, 488.4864502, -787.3532104, 783.6451416

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6491476, upper bound: 4810.6468278
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6491470, upper bound: 4810.6408013
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1641.2375488, 1869.1540527, -3286.6770020, 3595.8173828, -5237.0546875, 5155.8310547
1: -186.9884949, 123.8148651, -362.2013855, 250.7908936, -437.7793884, 486.0162354
2: -287.6911011, 349.7573853, -572.4530029, 675.2253418, -962.9164429, 922.2103882
3: -339.2583313, 217.7254944, -668.9405518, 423.4052429, -762.6634521, 886.6660156
4: -251.5863647, 283.4991150, -501.8109741, 545.7238770, -797.3101196, 785.3098755

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6484749, upper bound: 4810.6531868
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -4810.6457973, upper bound: 4810.6371741
time: 0.73 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.78 + 415.23 = 420.02 seconds
