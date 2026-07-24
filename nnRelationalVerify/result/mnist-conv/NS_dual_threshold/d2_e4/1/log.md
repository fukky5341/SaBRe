## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5346226540000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0611053, 1.0611053)
1: (-9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1834817, 1.1834817)
2: (-4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4875827, 1.4875827)
3: (-11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4724336, 1.4724336)
4: (-8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5915928, 1.5915928)
5: (-0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3831000, 1.3831000)
6: (5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248650, 1.2248650)
7: (-18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1333981, 1.1333976)
8: (-1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7780180, 1.7780180)
9: (-8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0668039, 1.0668039)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.67 + 33.40 = 56.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.5373092, upper bound: 0.5373085

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5328667
time: 3.64 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5373045
time: 3.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.69 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5328667
NS_B2, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5373045

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -4.7338657, -3.2512143, -4.7332306, -3.2545981, -1.0551682, 1.0583587
1: -9.6277485, -7.8912416, -9.6261415, -7.8926196, -1.1804724, 1.1797686
2: -4.8912487, -3.2926872, -4.8868947, -3.2939901, -1.4831171, 1.4809184
3: -11.5047359, -9.6225996, -11.5036144, -9.6244411, -1.4692516, 1.4702477
4: -8.0193853, -6.0287986, -8.0182896, -6.0331798, -1.5820498, 1.5853009
5: -0.4151219, 1.0413597, -0.4142461, 1.0373187, -1.3770638, 1.3801675
6: 5.8219867, 7.1746693, 5.8290596, 7.1746392, -1.2225838, 1.2157493
7: -18.3083210, -16.2119331, -18.3063316, -16.2130280, -1.1298628, 1.1292343
8: -1.0620036, 0.7289176, -1.0611186, 0.7249036, -1.7717285, 1.7737212
9: -8.3875837, -6.9192224, -8.3870287, -6.9252753, -1.0580244, 1.0632544

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5328673
time: 3.86 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5328667
time: 4.03 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -4.7340469, -3.2502491, -4.7620764, -3.2465339, -1.0692582, 1.0759754
1: -9.6282072, -7.8908520, -9.6346922, -7.8848681, -1.1927881, 1.1884336
2: -4.8924828, -3.2923195, -4.9031639, -3.2683783, -1.5042901, 1.4965181
3: -11.5050507, -9.6220722, -11.5137825, -9.6204147, -1.4736729, 1.4818850
4: -8.0196953, -6.0275483, -8.0546007, -6.0243802, -1.6094661, 1.6125636
5: -0.4153707, 1.0425110, -0.4412295, 1.0492529, -1.3895788, 1.3971329
6: 5.8199720, 7.1746774, 5.8082347, 7.2095165, -1.2326741, 1.2367029
7: -18.3088818, -16.2116222, -18.3103275, -16.1992168, -1.1438727, 1.1379371
8: -1.0622549, 0.7300658, -1.0775609, 0.7386322, -1.7881212, 1.7919273
9: -8.3877420, -6.9174995, -8.4240532, -6.9141598, -1.0701394, 1.0761955

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373057
time: 3.75 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373052
time: 4.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.95 seconds
NS_B1_A1, status: Status.VERIFIED, split count: 2, time: 29.95
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5328673
NS_B1_A2, status: Status.VERIFIED, split count: 2, time: 29.95
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5328667
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 29.95
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373057
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 29.95
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373052

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -4.7332306, -3.2545981, -4.7620764, -3.2465339, -1.0632668, 1.0703640
1: -9.6261415, -7.8926196, -9.6346922, -7.8848681, -1.1858778, 1.1861434
2: -4.8868947, -3.2939901, -4.9031639, -3.2683783, -1.4983354, 1.4933743
3: -11.5036144, -9.6244411, -11.5137825, -9.6204147, -1.4723144, 1.4790549
4: -8.0182896, -6.0331798, -8.0546007, -6.0243802, -1.5876770, 1.6039891
5: -0.4142461, 1.0373187, -0.4412295, 1.0492529, -1.3879757, 1.3914738
6: 5.8290596, 7.1746392, 5.8082347, 7.2095165, -1.2236152, 1.2374911
7: -18.3063316, -16.2130280, -18.3103275, -16.1992168, -1.1403289, 1.1319613
8: -1.0611186, 0.7249036, -1.0775609, 0.7386322, -1.7829814, 1.7863245
9: -8.3870287, -6.9252753, -8.4240532, -6.9141598, -1.0677843, 1.0677860

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317097, upper bound: 0.5370128
time: 3.79 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328641, upper bound: 0.5373012
time: 3.67 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -4.7620764, -3.2465339, -4.7620764, -3.2465339, -1.0749669, 1.0749671
1: -9.6346922, -7.8848681, -9.6346922, -7.8848681, -1.1984000, 1.1984000
2: -4.9031639, -3.2683783, -4.9031639, -3.2683783, -1.5028892, 1.5028887
3: -11.5137825, -9.6204147, -11.5137825, -9.6204147, -1.4794559, 1.4794559
4: -8.0546007, -6.0243802, -8.0546007, -6.0243802, -1.6160936, 1.6160936
5: -0.4412295, 1.0492529, -0.4412295, 1.0492529, -1.4035792, 1.4035158
6: 5.8082347, 7.2095165, 5.8082347, 7.2095165, -1.2445674, 1.2456303
7: -18.3103275, -16.1992168, -18.3103275, -16.1992168, -1.1414557, 1.1414557
8: -1.0775609, 0.7386322, -1.0775609, 0.7386322, -1.7919521, 1.7919521
9: -8.4240532, -6.9141598, -8.4240532, -6.9141598, -1.0730672, 1.0730677

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5325788, upper bound: 0.5317093
time: 7.46 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5328643, upper bound: 0.5328634
time: 3.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.93 seconds
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 6, lower bound: -0.5317097, upper bound: 0.5370128
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 32.93
Output dim: 6, lower bound: -0.5328641, upper bound: 0.5373012
NS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 32.93
Output dim: 6, lower bound: -0.5325788, upper bound: 0.5317093
NS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 32.93
Output dim: 6, lower bound: -0.5328643, upper bound: 0.5328634

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.7326021, -3.2571230, -4.7602196, -3.2515514, -1.0571318, 1.0644913
1: -9.6254997, -7.8942404, -9.6333408, -7.8880601, -1.1814694, 1.1825404
2: -4.8831668, -3.2949243, -4.8956671, -3.2714596, -1.4888430, 1.4840827
3: -11.5023117, -9.6255856, -11.5112305, -9.6228199, -1.4688835, 1.4751663
4: -8.0114708, -6.0339384, -8.0411758, -6.0279651, -1.5764513, 1.5903831
5: -0.4126616, 1.0346471, -0.4377365, 1.0444716, -1.3815804, 1.3840337
6: 5.8305645, 7.1701183, 5.8128748, 7.2007589, -1.2133956, 1.2285333
7: -18.3028908, -16.2142429, -18.3034668, -16.2020912, -1.1339717, 1.1244345
8: -1.0600471, 0.7232952, -1.0757608, 0.7353292, -1.7786212, 1.7825704
9: -8.3858986, -6.9260759, -8.4214706, -6.9155245, -1.0651507, 1.0646255

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5366373
time: 3.61 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5370123
time: 3.68 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.7332296, -3.2545986, -4.7620754, -3.2465360, -1.0611382, 1.0697687
1: -9.6261425, -7.8926210, -9.6346903, -7.8848705, -1.1858468, 1.1865931
2: -4.8868933, -3.2939906, -4.9031587, -3.2683787, -1.4972863, 1.4894691
3: -11.5036144, -9.6244392, -11.5137806, -9.6204147, -1.4738865, 1.4790521
4: -8.0182886, -6.0331807, -8.0545931, -6.0243816, -1.5870533, 1.5958638
5: -0.4142466, 1.0373173, -0.4412285, 1.0492508, -1.3870001, 1.3910561
6: 5.8290596, 7.1746364, 5.8082361, 7.2095103, -1.2217007, 1.2374887
7: -18.3063316, -16.2130280, -18.3103218, -16.1992188, -1.1399937, 1.1280074
8: -1.0611181, 0.7249026, -1.0775604, 0.7386298, -1.7854576, 1.7855392
9: -8.3870277, -6.9252758, -8.4240522, -6.9141612, -1.0674381, 1.0686159

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328636, upper bound: 0.5369258
time: 3.67 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328636, upper bound: 0.5373005
time: 3.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.27 seconds
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5366373
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5370123
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 6, lower bound: -0.5328636, upper bound: 0.5369258
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.27
Output dim: 6, lower bound: -0.5328636, upper bound: 0.5373005

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.7233272, -3.2710490, -4.7575655, -3.2587481, -1.0259056, 1.0482535
1: -9.5933523, -7.9076853, -9.6156769, -7.8899431, -1.1339116, 1.1521220
2: -4.8722439, -3.3239179, -4.8936863, -3.2868171, -1.4289799, 1.4509540
3: -11.4917555, -9.6323376, -11.5059185, -9.6246157, -1.4564295, 1.4611483
4: -8.0012541, -6.0458460, -8.0383406, -6.0345154, -1.5567842, 1.5828938
5: -0.4056897, 1.0275545, -0.4342482, 1.0416397, -1.3717704, 1.3066900
6: 5.8373804, 7.1668420, 5.8155851, 7.1991625, -1.1647401, 1.2213154
7: -18.2941151, -16.2248840, -18.3003674, -16.2078056, -1.1159286, 1.1102505
8: -1.0391774, 0.7131982, -1.0647917, 0.7331219, -1.7547317, 1.7221866
9: -8.3702011, -6.9550662, -8.4182825, -6.9315481, -0.9832201, 1.0315824

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317090, upper bound: 0.5357644
time: 3.88 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5366373
time: 3.68 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.7326026, -3.2571275, -4.7602186, -3.2515540, -1.0571294, 1.0560157
1: -9.6254959, -7.8942404, -9.6333380, -7.8880610, -1.1591232, 1.1803877
2: -4.8831654, -3.2949276, -4.8956676, -3.2714629, -1.4816518, 1.4664731
3: -11.5023079, -9.6255875, -11.5112305, -9.6228218, -1.4645114, 1.4751635
4: -8.0114689, -6.0339451, -8.0411739, -6.0279660, -1.5747838, 1.5851097
5: -0.4126581, 1.0346460, -0.4377357, 1.0444709, -1.3810253, 1.3822517
6: 5.8305645, 7.1701198, 5.8128748, 7.2007575, -1.2151227, 1.2274084
7: -18.3028908, -16.2142487, -18.3034649, -16.2020912, -1.1326585, 1.1202412
8: -1.0600410, 0.7232928, -1.0757575, 0.7353296, -1.7621675, 1.7770772
9: -8.3858986, -6.9260898, -8.4214706, -6.9155283, -1.0607667, 1.0367548

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5361394
time: 3.53 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5370123
time: 3.91 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.7238388, -3.2685215, -4.7594266, -3.2537274, -1.0303020, 1.0535336
1: -9.5939732, -7.9060612, -9.6170263, -7.8867555, -1.1381545, 1.1561766
2: -4.8757210, -3.3229818, -4.9011745, -3.2837367, -1.4365025, 1.4563432
3: -11.4930553, -9.6311922, -11.5084696, -9.6222067, -1.4614115, 1.4650388
4: -8.0080671, -6.0451837, -8.0517588, -6.0309386, -1.5673699, 1.5882912
5: -0.4072628, 1.0301102, -0.4377296, 1.0464220, -1.3771677, 1.3133593
6: 5.8360152, 7.1713619, 5.8109550, 7.2079144, -1.1735287, 1.2302611
7: -18.2975559, -16.2237225, -18.3072224, -16.2049408, -1.1219435, 1.1138029
8: -1.0402541, 0.7147098, -1.0665989, 0.7364221, -1.7615814, 1.7250605
9: -8.3712406, -6.9542661, -8.4208775, -6.9301863, -0.9857421, 1.0355661

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5326492, upper bound: 0.5349386
time: 3.58 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328607, upper bound: 0.5369228
time: 3.70 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.7332287, -3.2546029, -4.7620754, -3.2465377, -1.0611353, 1.0612946
1: -9.6261368, -7.8926201, -9.6346884, -7.8848701, -1.1635020, 1.1845257
2: -4.8868918, -3.2939935, -4.9031577, -3.2683814, -1.4900932, 1.4718595
3: -11.5036125, -9.6244421, -11.5137806, -9.6204157, -1.4695158, 1.4790492
4: -8.0182858, -6.0331855, -8.0545912, -6.0243826, -1.5853868, 1.5905704
5: -0.4142427, 1.0373170, -0.4412278, 1.0492493, -1.3864236, 1.3892741
6: 5.8290606, 7.1746354, 5.8082376, 7.2095089, -1.2234249, 1.2363629
7: -18.3063278, -16.2130356, -18.3103199, -16.1992207, -1.1386819, 1.1238112
8: -1.0611134, 0.7249026, -1.0775580, 0.7386308, -1.7690086, 1.7800012
9: -8.3870277, -6.9252872, -8.4240503, -6.9141665, -1.0630450, 1.0407457

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B2_A2_A1

### Relational analysis result of NS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5326492, upper bound: 0.5353137
time: 3.80 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2

### Relational analysis result of NS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328607, upper bound: 0.5372973
time: 3.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 29.22 seconds
NS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5317090, upper bound: 0.5357644
NS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5366373
NS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5361394
NS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5317091, upper bound: 0.5370123
NS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5326492, upper bound: 0.5349386
NS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5328607, upper bound: 0.5369228
NS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5326492, upper bound: 0.5353137
NS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 6, lower bound: -0.5328607, upper bound: 0.5372973

## BFS NS instance: NS_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -4.7222109, -3.2735348, -4.7575655, -3.2587481, -1.0247593, 1.0468318
1: -9.5926018, -7.9092584, -9.6156769, -7.8899431, -1.1330862, 1.1505170
2: -4.8687172, -3.3260641, -4.8936863, -3.2868171, -1.4274273, 1.4487104
3: -11.4905157, -9.6335878, -11.5059185, -9.6246157, -1.4548712, 1.4600539
4: -7.9947066, -6.0485764, -8.0383406, -6.0345154, -1.5509987, 1.5794125
5: -0.4037753, 1.0255468, -0.4342482, 1.0416397, -1.3693862, 1.3055127
6: 5.8404064, 7.1626062, 5.8155851, 7.1991625, -1.1616755, 1.2170701
7: -18.2906971, -16.2264862, -18.3003674, -16.2078056, -1.1128540, 1.1083469
8: -1.0384502, 0.7115955, -1.0647917, 0.7331219, -1.7533717, 1.7202950
9: -8.3688326, -6.9556513, -8.4182825, -6.9315481, -0.9817696, 1.0306742

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B1_A1_A1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5337738
time: 3.67 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5357611
time: 3.76 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -4.7238369, -3.2685230, -4.7575655, -3.2587481, -1.0265636, 1.0492513
1: -9.5939732, -7.9060631, -9.6156769, -7.8899431, -1.1344075, 1.1541216
2: -4.8757181, -3.3229835, -4.8936863, -3.2868171, -1.4295478, 1.4519711
3: -11.4930553, -9.6311913, -11.5059185, -9.6246157, -1.4577894, 1.4620981
4: -8.0080624, -6.0451841, -8.0383406, -6.0345154, -1.5635991, 1.5812297
5: -0.4072614, 1.0301083, -0.4342482, 1.0416397, -1.3734536, 1.3077590
6: 5.8360167, 7.1713576, 5.8155851, 7.1991625, -1.1646605, 1.2245107
7: -18.2975483, -16.2237225, -18.3003674, -16.2078056, -1.1191750, 1.1111135
8: -1.0402532, 0.7147083, -1.0647917, 0.7331219, -1.7553353, 1.7230406
9: -8.3712387, -6.9542670, -8.4182825, -6.9315481, -0.9840899, 1.0323215

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B1_A1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5346476
time: 3.94 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5366339
time: 4.14 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -4.7314105, -3.2596128, -4.7602186, -3.2515540, -1.0401545, 1.0545986
1: -9.6247311, -7.8958173, -9.6333380, -7.8880610, -1.1436279, 1.1789844
2: -4.8794761, -3.2970757, -4.8956676, -3.2714629, -1.4526501, 1.4642291
3: -11.5010662, -9.6268320, -11.5112305, -9.6228218, -1.4639306, 1.4740705
4: -8.0049238, -6.0367374, -8.0411739, -6.0279660, -1.5690098, 1.5884228
5: -0.4107330, 1.0325646, -0.4377357, 1.0444709, -1.3786454, 1.3156934
6: 5.8336821, 7.1658812, 5.8128748, 7.2007575, -1.1718402, 1.2231646
7: -18.2994709, -16.2158871, -18.3034649, -16.2020912, -1.1295815, 1.1185961
8: -1.0593195, 0.7216253, -1.0757575, 0.7353296, -1.7608037, 1.7446933
9: -8.3844852, -6.9266734, -8.4214706, -6.9155283, -1.0139966, 1.0358446

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B1_A2_A1_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5341489
time: 3.54 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_A2

### Relational analysis result of NS_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5361361
time: 3.70 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -4.7332277, -3.2546053, -4.7602186, -3.2515540, -1.0577054, 1.0570073
1: -9.6261368, -7.8926220, -9.6333380, -7.8880610, -1.1596634, 1.1816952
2: -4.8868885, -3.2939949, -4.8956676, -3.2714629, -1.4826560, 1.4674888
3: -11.5036106, -9.6244421, -11.5112305, -9.6228218, -1.4658856, 1.4761081
4: -8.0182819, -6.0331860, -8.0411739, -6.0279660, -1.5816026, 1.5834742
5: -0.4142418, 1.0373145, -0.4377357, 1.0444709, -1.3826962, 1.3834252
6: 5.8290606, 7.1746330, 5.8128748, 7.2007575, -1.2149963, 1.2303183
7: -18.3063240, -16.2130318, -18.3034649, -16.2020912, -1.1357906, 1.1210971
8: -1.0611134, 0.7249017, -1.0757575, 0.7353296, -1.7627606, 1.7777309
9: -8.3870258, -6.9252892, -8.4214706, -6.9155283, -1.0612640, 1.0374973

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B1_A2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5350218
time: 3.79 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5370089
time: 3.60 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -4.7220464, -3.2715216, -4.7589130, -3.2545838, -1.0267825, 1.0496981
1: -9.5866623, -7.9085054, -9.6149616, -7.8874402, -1.1298656, 1.1519494
2: -4.8696737, -3.3244746, -4.8994560, -3.2841606, -1.4294233, 1.4526200
3: -11.4896660, -9.6346836, -11.5075035, -9.6232042, -1.4560857, 1.4590087
4: -7.9986749, -6.0471601, -8.0490780, -6.0314994, -1.5572658, 1.5833402
5: -0.4054251, 1.0254929, -0.4372114, 1.0451090, -1.3703570, 1.3053198
6: 5.8389645, 7.1665354, 5.8117943, 7.2065434, -1.1685762, 1.2241626
7: -18.2832603, -16.2255516, -18.3031712, -16.2054558, -1.1071515, 1.1081619
8: -1.0382028, 0.7125206, -1.0659986, 0.7357965, -1.7575264, 1.7205386
9: -8.3644342, -6.9556260, -8.4189472, -6.9305658, -0.9784184, 1.0323174

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5349368
time: 3.75 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5349375
time: 3.68 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -4.7326913, -3.2593021, -4.7594213, -3.2537367, -1.0416174, 1.0643265
1: -9.5977383, -7.8699961, -9.6170111, -7.8867593, -1.1424360, 1.1779602
2: -4.8877640, -3.3018885, -4.9011631, -3.2837424, -1.4470134, 1.4788127
3: -11.5091286, -9.6273499, -11.5084600, -9.6222172, -1.4859247, 1.4671044
4: -8.0251055, -6.0176835, -8.0517321, -6.0309420, -1.5820851, 1.6008697
5: -0.4171944, 1.0336285, -0.4377252, 1.0464089, -1.3957577, 1.3163800
6: 5.8106351, 7.1742682, 5.8109627, 7.2079077, -1.1895089, 1.2332797
7: -18.3003445, -16.1527691, -18.3071880, -16.2049446, -1.1236610, 1.1468949
8: -1.0535822, 0.7173719, -1.0665936, 0.7364168, -1.7783098, 1.7294564
9: -8.3869658, -6.9266863, -8.4208632, -6.9301901, -0.9998260, 1.0445721

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5367116
time: 3.54 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5369229
time: 3.52 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -4.7314243, -3.2575998, -4.7615633, -3.2473917, -1.0581522, 1.0574462
1: -9.6188345, -7.8950682, -9.6326237, -7.8855596, -1.1552107, 1.1800249
2: -4.8808098, -3.2954893, -4.9014387, -3.2688031, -1.4831972, 1.4681387
3: -11.5002270, -9.6279364, -11.5128174, -9.6214123, -1.4642196, 1.4730148
4: -8.0088892, -6.0351739, -8.0519123, -6.0249443, -1.5752854, 1.5855083
5: -0.4124000, 1.0327090, -0.4407088, 1.0479441, -1.3796272, 1.3804674
6: 5.8320398, 7.1698093, 5.8090777, 7.2081385, -1.2187181, 1.2302620
7: -18.2920341, -16.2148571, -18.3062687, -16.1997356, -1.1238804, 1.1181650
8: -1.0590429, 0.7226973, -1.0769558, 0.7380037, -1.7649555, 1.7751241
9: -8.3802204, -6.9266453, -8.4221201, -6.9145441, -1.0557680, 1.0375001

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5353117
time: 3.85 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5353114
time: 3.81 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -4.7427163, -3.2454362, -4.7620716, -3.2465463, -1.0734115, 1.0719976
1: -9.6300669, -7.8565602, -9.6346722, -7.8848777, -1.1679492, 1.2055769
2: -4.9002476, -3.2729313, -4.9031472, -3.2683856, -1.5020723, 1.4932184
3: -11.5197754, -9.6206064, -11.5137711, -9.6204262, -1.4940085, 1.4810991
4: -8.0353584, -6.0051813, -8.0545683, -6.0243883, -1.6001372, 1.6032910
5: -0.4241232, 1.0414729, -0.4412222, 1.0492365, -1.4043112, 1.3920708
6: 5.8030343, 7.1775417, 5.8082447, 7.2095032, -1.2395616, 1.2393801
7: -18.3091221, -16.1417847, -18.3102856, -16.1992264, -1.1403708, 1.1575460
8: -1.0744891, 0.7281008, -1.0775528, 0.7386270, -1.7858558, 1.7843337
9: -8.4031992, -6.8977060, -8.4240370, -6.9141703, -1.0772653, 1.0497620

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of NS_B2_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5370863
time: 3.79 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5370862
time: 3.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 29.43 seconds
NS_B2_A1_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5337738
NS_B2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5357611
NS_B2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5346476
NS_B2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5366339
NS_B2_A1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5341489
NS_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5361361
NS_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5314909, upper bound: 0.5350218
NS_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5317061, upper bound: 0.5370089
NS_B2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5349368
NS_B2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5349375
NS_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5367116
NS_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5369229
NS_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5353117
NS_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5353114
NS_B2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5370863
NS_B2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 29.43
Output dim: 6, lower bound: -0.5308748, upper bound: 0.5370862

## BFS NS instance: NS_B2_A1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -4.7310705, -3.2643247, -4.7575603, -3.2587543, -1.0360770, 1.0576172
1: -9.5963364, -7.8731866, -9.6156607, -7.8899488, -1.1373413, 1.1724148
2: -4.8807607, -3.3049731, -4.8936749, -3.2868209, -1.4379416, 1.4711862
3: -11.5065899, -9.6297560, -11.5059099, -9.6246262, -1.4794021, 1.4621134
4: -8.0117426, -6.0210772, -8.0383167, -6.0345197, -1.5657005, 1.5919905
5: -0.4137297, 1.0290602, -0.4342439, 1.0416274, -1.3880606, 1.3085299
6: 5.8150072, 7.1655130, 5.8155928, 7.1991553, -1.1775928, 1.2200909
7: -18.2934875, -16.1555271, -18.3003330, -16.2078114, -1.1145597, 1.1415424
8: -1.0517793, 0.7142520, -1.0647869, 0.7331157, -1.7701058, 1.7246838
9: -8.3845930, -6.9280682, -8.4182663, -6.9315510, -0.9958692, 1.0396762

Time for backsubstitution: 21.65 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.07 + 560.26 = 616.34 seconds
