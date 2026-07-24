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
execution time: IAR + RelationalAnalysis = 21.87 + 33.01 = 54.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1674431, upper bound: 0.1674431

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 5792
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5846

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1647950, upper bound: 0.1674393
time: 5.13 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674392, upper bound: 0.1674393
time: 4.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.24
Output dim: 7, lower bound: -0.1647950, upper bound: 0.1674393
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.24
Output dim: 7, lower bound: -0.1674392, upper bound: 0.1674393

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -16.2240829, -14.4295721, -16.2307568, -14.4270382, -0.5328052, 0.5366604
1: -6.1880655, -5.3179178, -6.1902418, -5.3172083, -0.3534091, 0.3548272
2: -7.2550921, -6.0331717, -7.2562399, -6.0330191, -0.4591348, 0.4605234
3: -2.2775085, -1.0416799, -2.2900450, -1.0337157, -0.4628158, 0.4680443
4: -5.5545397, -3.9123311, -5.5600314, -3.8999250, -0.5351784, 0.5273253
5: -9.1476173, -7.8029943, -9.1574631, -7.7972879, -0.4801238, 0.4846215
6: -15.1147470, -13.7985783, -15.1197796, -13.7964497, -0.3044527, 0.3069728
7: 4.1345277, 5.1048341, 4.1310391, 5.1148396, -0.2896101, 0.2824893
8: -4.9942856, -3.8699722, -5.0014629, -3.8667231, -0.4067221, 0.4112172
9: -3.2980530, -2.0644577, -3.3007705, -2.0578520, -0.3707985, 0.3663129

Time for backsubstitution: 19.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 117

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5846

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1647949
time: 3.35 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1674394
time: 3.31 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -16.2318573, -14.4231892, -16.2318554, -14.4231911, -0.5442522, 0.5382044
1: -6.1904578, -5.3161325, -6.1904583, -5.3161325, -0.3568199, 0.3543539
2: -7.2571921, -6.0329413, -7.2571917, -6.0329428, -0.4629273, 0.4604368
3: -2.2900820, -1.0217257, -2.2900815, -1.0217230, -0.4791372, 0.4613595
4: -5.5683045, -3.8994555, -5.5683041, -3.8994551, -0.5301192, 0.5435613
5: -9.1575260, -7.7887836, -9.1575260, -7.7887812, -0.4930073, 0.4789248
6: -15.1201572, -13.7932272, -15.1201601, -13.7932253, -0.3130066, 0.3056903
7: 4.1257458, 5.1152573, 4.1257429, 5.1152577, -0.2860792, 0.2945769
8: -5.0022497, -3.8621655, -5.0022497, -3.8621631, -0.4179674, 0.4106336
9: -3.3048902, -2.0574994, -3.3048913, -2.0574975, -0.3682857, 0.3750532

Time for backsubstitution: 20.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 117

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5846

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674395, upper bound: 0.1647948
time: 6.88 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674395, upper bound: 0.1674394
time: 2.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.67 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.67
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1647949
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.67
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1674394
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.67
Output dim: 7, lower bound: -0.1674395, upper bound: 0.1647948
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.67
Output dim: 7, lower bound: -0.1674395, upper bound: 0.1674394

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -16.2240829, -14.4295721, -16.2316303, -14.4231892, -0.5366523, 0.5376601
1: -6.1880655, -5.3179178, -6.1904449, -5.3161325, -0.3544855, 0.3550203
2: -7.2550921, -6.0331717, -7.2569017, -6.0329456, -0.4582167, 0.4599650
3: -2.2775085, -1.0416799, -2.2900820, -1.0219121, -0.4637632, 0.4592086
4: -5.5545397, -3.9123311, -5.5681682, -3.8994555, -0.5298450, 0.5290251
5: -9.1476173, -7.8029943, -9.1575212, -7.7888513, -0.4811851, 0.4788148
6: -15.1147470, -13.7985783, -15.1201296, -13.7932272, -0.3070514, 0.3076205
7: 4.1345277, 5.1048341, 4.1257458, 5.1152472, -0.2857730, 0.2835722
8: -4.9942856, -3.8699722, -5.0022497, -3.8623199, -0.4091101, 0.4102117
9: -3.2980530, -2.0644577, -3.3048148, -2.0574994, -0.3682472, 0.3673125

Time for backsubstitution: 20.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5792
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5792

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1647947, upper bound: 0.1670913
time: 5.16 seconds

## Relational analysis of NS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1628994, upper bound: 0.1674393
time: 3.38 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1647948, upper bound: 0.1674392
time: 4.12 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -16.2316303, -14.4231892, -16.2240829, -14.4295721, -0.5376601, 0.5366521
1: -6.1904449, -5.3161325, -6.1880655, -5.3179178, -0.3550203, 0.3544854
2: -7.2569017, -6.0329456, -7.2550921, -6.0331717, -0.4599650, 0.4582167
3: -2.2900820, -1.0219121, -2.2775085, -1.0416799, -0.4592087, 0.4637632
4: -5.5681682, -3.8994555, -5.5545397, -3.9123311, -0.5290252, 0.5298451
5: -9.1575212, -7.7888513, -9.1476173, -7.8029943, -0.4788150, 0.4811851
6: -15.1201296, -13.7932272, -15.1147470, -13.7985783, -0.3076205, 0.3070514
7: 4.1257458, 5.1152472, 4.1345277, 5.1048341, -0.2835722, 0.2857731
8: -5.0022497, -3.8623199, -4.9942856, -3.8699722, -0.4102117, 0.4091101
9: -3.3048148, -2.0574994, -3.2980530, -2.0644577, -0.3673124, 0.3682472

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5792
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5792

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674380, upper bound: 0.1644469
time: 3.08 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674380, upper bound: 0.1647946
time: 3.94 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -16.2318573, -14.4231892, -16.2318573, -14.4231892, -0.5382042, 0.5382044
1: -6.1904578, -5.3161325, -6.1904578, -5.3161325, -0.3543537, 0.3543538
2: -7.2571921, -6.0329413, -7.2571921, -6.0329413, -0.4629276, 0.4629276
3: -2.2900820, -1.0217257, -2.2900820, -1.0217257, -0.4613590, 0.4613593
4: -5.5683045, -3.8994555, -5.5683045, -3.8994555, -0.5301189, 0.5301188
5: -9.1575260, -7.7887836, -9.1575260, -7.7887836, -0.4789250, 0.4789250
6: -15.1201572, -13.7932272, -15.1201572, -13.7932272, -0.3056899, 0.3056900
7: 4.1257458, 5.1152573, 4.1257458, 5.1152573, -0.2860787, 0.2860787
8: -5.0022497, -3.8621655, -5.0022497, -3.8621655, -0.4106328, 0.4106327
9: -3.3048902, -2.0574994, -3.3048902, -2.0574994, -0.3682857, 0.3682857

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5792
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5792

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674392, upper bound: 0.1644468
time: 4.21 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674392, upper bound: 0.1647946
time: 3.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.01 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -0.1628994, upper bound: 0.1674393
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -0.1647948, upper bound: 0.1674392
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -0.1674380, upper bound: 0.1644469
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -0.1674380, upper bound: 0.1647946
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -0.1674392, upper bound: 0.1644468
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.01
Output dim: 7, lower bound: -0.1674392, upper bound: 0.1647946

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16.2230434, -14.4305716, -16.2314587, -14.4233952, -0.5356829, 0.5366607
1: -6.1875868, -5.3279810, -6.1904364, -5.3182039, -0.3473464, 0.3451816
2: -7.2527461, -6.0789366, -7.2568049, -6.0423183, -0.4189800, 0.4141946
3: -2.2593853, -1.0424645, -2.2863691, -1.0219245, -0.4458852, 0.4430476
4: -5.5539689, -3.9261432, -5.5681601, -3.9022844, -0.5172992, 0.5152329
5: -9.0971804, -7.8050513, -9.1471910, -7.7888556, -0.4312763, 0.4334621
6: -15.1140747, -13.8040075, -15.1200485, -13.7943354, -0.3020204, 0.3021801
7: 4.1494269, 5.1041727, 4.1287999, 5.1152368, -0.2709001, 0.2700592
8: -4.9908805, -3.8708277, -5.0015492, -3.8624268, -0.4057784, 0.4067979
9: -3.2798357, -2.0652018, -3.3010800, -2.0575039, -0.3501503, 0.3508837

Time for backsubstitution: 20.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 117

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5792

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1628994, upper bound: 0.1655427
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1628994, upper bound: 0.1674380
time: 4.99 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.2240829, -14.4295759, -16.2316303, -14.4231892, -0.5366485, 0.5371008
1: -6.1880651, -5.3179173, -6.1904449, -5.3161325, -0.3544812, 0.3462496
2: -7.2550941, -6.0331707, -7.2569017, -6.0329456, -0.4582169, 0.4186656
3: -2.2775102, -1.0416794, -2.2900820, -1.0219121, -0.4476838, 0.4584693
4: -5.5545397, -3.9123328, -5.5681682, -3.8994555, -0.5292902, 0.5165159
5: -9.1476154, -7.8029943, -9.1575212, -7.7888513, -0.4360015, 0.4767773
6: -15.1147461, -13.7985764, -15.1201296, -13.7932272, -0.3068148, 0.3027214
7: 4.1345277, 5.1048341, 4.1257458, 5.1152472, -0.2723176, 0.2829634
8: -4.9942851, -3.8699718, -5.0022497, -3.8623199, -0.4061835, 0.4100173
9: -3.2980533, -2.0644577, -3.3048148, -2.0574994, -0.3518922, 0.3665722

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5792

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1647940, upper bound: 0.1655426
time: 3.08 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1655438
time: 4.57 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16.2129517, -14.4483919, -16.2239933, -14.4448042, -0.5006116, 0.5113800
1: -6.1869926, -5.3191409, -6.1859751, -5.3179736, -0.3515146, 0.3493716
2: -7.2475166, -6.0459871, -7.2550921, -6.0410399, -0.4426906, 0.4451847
3: -2.2744403, -1.0334294, -2.2680552, -1.0417252, -0.4435303, 0.4399111
4: -5.5534387, -3.9193039, -5.5544653, -3.9243271, -0.4986619, 0.5099437
5: -9.1415844, -7.8003364, -9.1379852, -7.8029962, -0.4628658, 0.4571627
6: -15.1192627, -13.7938337, -15.1142979, -13.7985783, -0.3067467, 0.3059231
7: 4.1272330, 5.1141601, 4.1346903, 5.1041775, -0.2811790, 0.2846754
8: -4.9973774, -3.8659668, -4.9913402, -3.8699942, -0.4053199, 0.4016101
9: -3.2929571, -2.0732853, -3.2979777, -2.0739970, -0.3429959, 0.3524101

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 117

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1625512
time: 3.19 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1644468
time: 3.90 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16.2316303, -14.4232025, -16.2240829, -14.4295807, -0.5207387, 0.4986219
1: -6.1904421, -5.3161330, -6.1880636, -5.3179178, -0.3498080, 0.3536997
2: -7.2569017, -6.0329514, -7.2550921, -6.0331755, -0.4530294, 0.4386744
3: -2.2900686, -1.0219131, -2.2775016, -1.0416801, -0.4355552, 0.4523706
4: -5.5681682, -3.8994708, -5.5545411, -3.9123406, -0.5145301, 0.4998090
5: -9.1575146, -7.7888513, -9.1476135, -7.8029943, -0.4548033, 0.4696878
6: -15.1201267, -13.7932272, -15.1147451, -13.7985783, -0.3066932, 0.3064409
7: 4.1257467, 5.1152468, 4.1345272, 5.1048350, -0.2824979, 0.2839290
8: -5.0022435, -3.8623199, -4.9942822, -3.8699703, -0.4028220, 0.4055333
9: -3.3048139, -2.0575137, -3.2980516, -2.0644670, -0.3557193, 0.3443075

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1628990
time: 3.03 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1647944
time: 3.53 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.2131729, -14.4483919, -16.2317657, -14.4384232, -0.5042512, 0.5129323
1: -6.1870060, -5.3191409, -6.1883688, -5.3161898, -0.3508470, 0.3492395
2: -7.2478042, -6.0459838, -7.2571921, -6.0408082, -0.4450505, 0.4498951
3: -2.2744403, -1.0332408, -2.2806287, -1.0217655, -0.4456713, 0.4403422
4: -5.5535746, -3.9193039, -5.5682240, -3.9114537, -0.5027905, 0.5102147
5: -9.1415901, -7.8002725, -9.1478939, -7.7887855, -0.4629867, 0.4577997
6: -15.1192932, -13.7938337, -15.1197119, -13.7932272, -0.3048165, 0.3045616
7: 4.1272330, 5.1141672, 4.1259089, 5.1145997, -0.2838850, 0.2849813
8: -4.9973774, -3.8658142, -4.9993048, -3.8621907, -0.4057438, 0.4040222
9: -3.2930343, -2.0732841, -3.3048158, -2.0670385, -0.3455465, 0.3524363

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 117

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674388, upper bound: 0.1625512
time: 3.16 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674391, upper bound: 0.1644467
time: 3.21 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.2318592, -14.4232025, -16.2318573, -14.4232006, -0.5251791, 0.5002432
1: -6.1904545, -5.3161330, -6.1904564, -5.3161330, -0.3491409, 0.3543525
2: -7.2571921, -6.0329461, -7.2571921, -6.0329447, -0.4552658, 0.4433844
3: -2.2900686, -1.0217247, -2.2900751, -1.0217242, -0.4377759, 0.4536231
4: -5.5683050, -3.8994708, -5.5683017, -3.8994651, -0.5186586, 0.5002193
5: -9.1575184, -7.7887836, -9.1575203, -7.7887836, -0.4549367, 0.4707066
6: -15.1201553, -13.7932272, -15.1201563, -13.7932272, -0.3047626, 0.3056894
7: 4.1257467, 5.1152558, 4.1257458, 5.1152582, -0.2860785, 0.2844391
8: -5.0022435, -3.8621659, -5.0022469, -3.8621659, -0.4032905, 0.4097842
9: -3.3048899, -2.0575132, -3.3048899, -2.0575085, -0.3582697, 0.3444821

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 5792
type: B, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674389, upper bound: 0.1628988
time: 5.99 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1674391, upper bound: 0.1647944
time: 3.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.57 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1628994, upper bound: 0.1655427
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1628994, upper bound: 0.1674380
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1647940, upper bound: 0.1655426
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1647949, upper bound: 0.1655438
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1625512
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1644468
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1628990
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674379, upper bound: 0.1647944
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674388, upper bound: 0.1625512
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674391, upper bound: 0.1644467
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674389, upper bound: 0.1628988
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.57
Output dim: 7, lower bound: -0.1674391, upper bound: 0.1647944

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.2230434, -14.4305716, -16.2316284, -14.4231930, -0.5356830, 0.5366631
1: -6.1875868, -5.3279810, -6.1904449, -5.3161321, -0.3473470, 0.3451952
2: -7.2527461, -6.0789366, -7.2569027, -6.0329437, -0.4189798, 0.4141994
3: -2.2593853, -1.0424645, -2.2900813, -1.0219116, -0.4451463, 0.4430757
4: -5.5539689, -3.9261432, -5.5681682, -3.8994572, -0.5173059, 0.5146829
5: -9.0971804, -7.8050513, -9.1575222, -7.7888513, -0.4292437, 0.4334809
6: -15.1140747, -13.8040075, -15.1201286, -13.7932262, -0.3020204, 0.3021056
7: 4.1494269, 5.1041727, 4.1257467, 5.1152472, -0.2702932, 0.2700658
8: -4.9908805, -3.8708277, -5.0022497, -3.8623195, -0.4056326, 0.4068007
9: -3.2798357, -2.0652018, -3.3048124, -2.0574987, -0.3494123, 0.3508953

Time for backsubstitution: 20.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5792
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5792

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.2127762, -14.4485941, -16.2229538, -14.4458008, -0.4996129, 0.5104262
1: -6.1869817, -5.3212099, -6.1854963, -5.3280392, -0.3416748, 0.3415985
2: -7.2474194, -6.0553584, -7.2527461, -6.0868044, -0.3969204, 0.4059253
3: -2.2707274, -1.0334401, -2.2499313, -1.0425062, -0.4273691, 0.4220330
4: -5.5534325, -3.9221311, -5.5538902, -3.9381371, -0.4848694, 0.4973980
5: -9.1312542, -7.8003445, -9.0875473, -7.8050528, -0.4175129, 0.4072541
6: -15.1191807, -13.7949467, -15.1136284, -13.8040075, -0.3013067, 0.3008921
7: 4.1302857, 5.1141453, 4.1495886, 5.1035147, -0.2676660, 0.2698025
8: -4.9966793, -3.8660746, -4.9879365, -3.8708515, -0.4019065, 0.3982787
9: -3.2892222, -2.0732892, -3.2797604, -2.0747437, -0.3265672, 0.3343133

Time for backsubstitution: 20.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1625513
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1625514
time: 2.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.2129517, -14.4483919, -16.2239933, -14.4448032, -0.5000519, 0.5113800
1: -6.1869926, -5.3191409, -6.1859751, -5.3179750, -0.3427432, 0.3493675
2: -7.2475166, -6.0459871, -7.2550941, -6.0410380, -0.4013908, 0.4451849
3: -2.2744403, -1.0334294, -2.2680564, -1.0417233, -0.4427907, 0.4238313
4: -5.5534387, -3.9193039, -5.5544653, -3.9243269, -0.4861524, 0.5093889
5: -9.1415844, -7.8003364, -9.1379852, -7.8029962, -0.4608283, 0.4119792
6: -15.1192627, -13.7938337, -15.1142969, -13.7985764, -0.3018480, 0.3056863
7: 4.1272330, 5.1141601, 4.1346912, 5.1041775, -0.2805701, 0.2712199
8: -4.9973774, -3.8659668, -4.9913421, -3.8699970, -0.4051256, 0.3986835
9: -3.2929571, -2.0732853, -3.2979779, -2.0739970, -0.3422555, 0.3360552

Time for backsubstitution: 20.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1644468
time: 2.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1644467
time: 3.46 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.2314568, -14.4234085, -16.2230434, -14.4305811, -0.5197399, 0.4975723
1: -6.1904316, -5.3182054, -6.1875839, -5.3279810, -0.3399687, 0.3446416
2: -7.2568049, -6.0423231, -7.2527461, -6.0789413, -0.4072471, 0.3993767
3: -2.2863555, -1.0219233, -2.2593780, -1.0424633, -0.4193943, 0.4344922
4: -5.5681601, -3.9022980, -5.5539680, -3.9261508, -0.5007378, 0.4872633
5: -9.1471844, -7.7888556, -9.0971756, -7.8050513, -0.4094505, 0.4197787
6: -15.1200485, -13.7943354, -15.1140747, -13.8040075, -0.3012528, 0.3014098
7: 4.1287975, 5.1152353, 4.1494250, 5.1041732, -0.2689849, 0.2690562
8: -5.0015469, -3.8624258, -4.9908791, -3.8708267, -0.3994086, 0.4022018
9: -3.3010781, -2.0575182, -3.2798345, -2.0652111, -0.3392904, 0.3262108

Time for backsubstitution: 20.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1628991
time: 3.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1628989
time: 3.38 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.2316303, -14.4232025, -16.2240829, -14.4295797, -0.5201791, 0.4985377
1: -6.1904421, -5.3161330, -6.1880631, -5.3179183, -0.3410366, 0.3532877
2: -7.2569017, -6.0329514, -7.2550941, -6.0331764, -0.4117129, 0.4386747
3: -2.2900686, -1.0219131, -2.2775030, -1.0416799, -0.4348159, 0.4362911
4: -5.5681682, -3.8994708, -5.5545406, -3.9123385, -0.5020206, 0.4992541
5: -9.1575146, -7.7888513, -9.1476135, -7.8029943, -0.4527656, 0.4245042
6: -15.1201267, -13.7932272, -15.1147451, -13.7985764, -0.3017942, 0.3062043
7: 4.1257467, 5.1152468, 4.1345272, 5.1048350, -0.2818890, 0.2704736
8: -5.0022435, -3.8623199, -4.9942822, -3.8699722, -0.4026277, 0.4026066
9: -3.3048139, -2.0575137, -3.2980511, -2.0644667, -0.3549790, 0.3279525

Time for backsubstitution: 20.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1647946
time: 2.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655424, upper bound: 0.1647946
time: 3.43 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16.2130051, -14.4485941, -16.2307243, -14.4394169, -0.5032523, 0.5125517
1: -6.1869946, -5.3212099, -6.1878901, -5.3262548, -0.3410077, 0.3426049
2: -7.2477102, -6.0553541, -7.2548447, -6.0865736, -0.3992682, 0.4107730
3: -2.2707274, -1.0332494, -2.2625065, -1.0225487, -0.4335666, 0.4224334
4: -5.5535665, -3.9221311, -5.5676527, -3.9252596, -0.4889982, 0.5014937
5: -9.1312580, -7.8002806, -9.0974579, -7.7908440, -0.4212477, 0.4078896
6: -15.1192131, -13.7949467, -15.1190443, -13.7986565, -0.2993763, 0.3030187
7: 4.1302857, 5.1141562, 4.1408081, 5.1139374, -0.2717935, 0.2701069
8: -4.9966793, -3.8659215, -4.9959011, -3.8630476, -0.4045383, 0.4007189
9: -3.2893023, -2.0732889, -3.2866004, -2.0677857, -0.3291178, 0.3343440

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655435, upper bound: 0.1625512
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655435, upper bound: 0.1625512
time: 4.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.2131729, -14.4483919, -16.2317657, -14.4384212, -0.5036913, 0.5129323
1: -6.1870060, -5.3191409, -6.1883678, -5.3161907, -0.3420758, 0.3492355
2: -7.2478042, -6.0459838, -7.2571921, -6.0408072, -0.4037342, 0.4498951
3: -2.2744403, -1.0332408, -2.2806292, -1.0217671, -0.4456720, 0.4242709
4: -5.5535746, -3.9193039, -5.5682240, -3.9114513, -0.4902811, 0.5102147
5: -9.1415901, -7.8002725, -9.1478920, -7.7887855, -0.4629862, 0.4126544
6: -15.1192932, -13.7938337, -15.1197119, -13.7932262, -0.2999176, 0.3045616
7: 4.1272330, 5.1141672, 4.1259093, 5.1145997, -0.2838850, 0.2715391
8: -4.9973774, -3.8658142, -4.9993057, -3.8621888, -0.4057438, 0.4011182
9: -3.2930343, -2.0732841, -3.3048172, -2.0670388, -0.3448062, 0.3360919

Time for backsubstitution: 20.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 522

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655433, upper bound: 0.1644467
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.1655433, upper bound: 0.1644467
time: 4.96 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16.2316875, -14.4234085, -16.2308197, -14.4241943, -0.5241801, 0.4998628
1: -6.1904445, -5.3182054, -6.1899772, -5.3261976, -0.3393016, 0.3456479
2: -7.2570972, -6.0423169, -7.2548447, -6.0787106, -0.4094832, 0.4042535
3: -2.2863555, -1.0217328, -2.2719493, -1.0225074, -0.4256668, 0.4357448
4: -5.5682983, -3.9022980, -5.5677309, -3.9132733, -0.5048664, 0.4914836
5: -9.1471882, -7.7887940, -9.1070862, -7.7908401, -0.4131892, 0.4207981
6: -15.1200752, -13.7943354, -15.1194916, -13.7986565, -0.2993222, 0.3035366
7: 4.1287975, 5.1152468, 4.1406460, 5.1145959, -0.2731123, 0.2695647
8: -5.0015469, -3.8622723, -4.9988418, -3.8630209, -0.4020847, 0.4064527
9: -3.3011580, -2.0575180, -3.2866724, -2.0582535, -0.3418410, 0.3263896

Time for backsubstitution: 20.79 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.88 + 553.36 = 608.24 seconds
