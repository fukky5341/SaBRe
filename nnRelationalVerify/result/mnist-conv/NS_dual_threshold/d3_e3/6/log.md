## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.566640958


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967296, 1.2967298)
1: (-2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4835277, 1.4835272)
2: (1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2275105, 1.2275107)
3: (-6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0011253, 1.0011251)
4: (-2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9452736, 0.9452736)
5: (-4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0601525, 1.0601530)
6: (-4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4067559, 1.4067559)
7: (-8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8539648, 0.8539647)
8: (-4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4132328, 1.4132328)
9: (-11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9780154, 0.9780157)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.71 + 34.44 = 57.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.5694897, upper bound: 0.5694879

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693774, upper bound: 0.5665206
time: 5.04 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693739, upper bound: 0.5693724
time: 4.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.00
Output dim: 2, lower bound: -0.5693774, upper bound: 0.5665206
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.00
Output dim: 2, lower bound: -0.5693739, upper bound: 0.5693724

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.4845095, -8.6201115, -10.4847822, -8.6198349, -1.2962132, 1.2960172
1: -2.9913936, -1.2639763, -2.9937477, -1.2634897, -1.4809794, 1.4830422
2: 1.9922096, 3.3791320, 1.9920239, 3.3824191, -1.2273016, 1.2242949
3: -6.9479208, -5.5200839, -6.9481716, -5.5183735, -1.0008707, 0.9992852
4: -2.0611091, -0.6649556, -2.0612500, -0.6644926, -0.9449649, 0.9445646
5: -4.3437881, -3.0172141, -4.3456483, -3.0169828, -1.0583272, 1.0599289
6: -4.3318205, -2.5348814, -4.3337641, -2.5347354, -1.4043903, 1.4064388
7: -8.5212173, -7.2031975, -8.5212250, -7.2025671, -0.8535166, 0.8529903
8: -4.3130350, -2.7289605, -4.3148742, -2.7288890, -1.4111795, 1.4131427
9: -11.8572035, -10.1522961, -11.8572426, -10.1520758, -0.9777031, 0.9774911

Time for backsubstitution: 20.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5665213, upper bound: 0.5665210
time: 6.14 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5665199
time: 6.74 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.5011539, -8.6151848, -10.4847765, -8.6198387, -1.3162313, 1.3138661
1: -3.0011899, -1.2012343, -2.9937375, -1.2634931, -1.5023131, 1.5455451
2: 1.9156601, 3.3923306, 1.9920256, 3.3823879, -1.2636483, 1.2448940
3: -6.9948902, -5.5134830, -6.9481707, -5.5183849, -1.0490787, 1.0167339
4: -2.0803468, -0.6623344, -2.0612493, -0.6644950, -0.9645107, 0.9536990
5: -4.3502407, -2.9686344, -4.3456340, -3.0169847, -1.0703800, 1.1006465
6: -4.3489075, -2.4950233, -4.3337488, -2.5347362, -1.4435649, 1.4262528
7: -8.5299654, -7.1895690, -8.5212240, -7.2025723, -0.8738523, 0.8643689
8: -4.3281984, -2.6837430, -4.3148580, -2.7288902, -1.4218879, 1.4591913
9: -11.8633480, -10.1493073, -11.8572416, -10.1520786, -0.9842935, 0.9896301

Time for backsubstitution: 20.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693720
time: 4.97 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693713
time: 5.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.92 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 30.92
Output dim: 2, lower bound: -0.5665213, upper bound: 0.5665210
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 30.92
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5665199
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.92
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693720
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.92
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693713

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.5007982, -8.6151876, -10.4845095, -8.6201115, -1.3151193, 1.3026605
1: -3.0011857, -1.2013631, -2.9913936, -1.2639763, -1.4892454, 1.5429053
2: 1.9156780, 3.3922989, 1.9922096, 3.3791320, -1.2603788, 1.2392135
3: -6.9948220, -5.5135183, -6.9479208, -5.5200839, -1.0471609, 1.0075488
4: -2.0801384, -0.6623397, -2.0611091, -0.6649556, -0.9635644, 0.9472473
5: -4.3501863, -2.9686558, -4.3437881, -3.0172141, -1.0663528, 1.0987835
6: -4.3488178, -2.4950233, -4.3318205, -2.5348814, -1.4243460, 1.4238801
7: -8.5299644, -7.1897507, -8.5212173, -7.2031975, -0.8622465, 0.8638297
8: -4.3281212, -2.6837451, -4.3130350, -2.7289605, -1.4207597, 1.4571233
9: -11.8632212, -10.1493111, -11.8572035, -10.1522961, -0.9836352, 0.9800758

Time for backsubstitution: 20.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5664689, upper bound: 0.5646594
time: 5.50 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665197, upper bound: 0.5693686
time: 5.11 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.5031013, -8.6151762, -10.5031013, -8.6151762, -1.3286009, 1.3286011
1: -3.0012190, -1.2005267, -3.0012190, -1.2005267, -1.5190291, 1.5190296
2: 1.9155622, 3.3925014, 1.9155622, 3.3925014, -1.2818947, 1.2765205
3: -6.9952583, -5.5132847, -6.9952583, -5.5132847, -1.0312703, 1.0312703
4: -2.0814836, -0.6623116, -2.0814836, -0.6623116, -0.9634238, 0.9634240
5: -4.3505349, -2.9685144, -4.3505349, -2.9685144, -1.0925803, 1.0925801
6: -4.3493891, -2.4950237, -4.3493891, -2.4950237, -1.4458604, 1.4497328
7: -8.5299635, -7.1885777, -8.5299635, -7.1885777, -0.8852117, 0.8852117
8: -4.3286333, -2.6837440, -4.3286333, -2.6837440, -1.4606924, 1.4606924
9: -11.8640461, -10.1492825, -11.8640461, -10.1492825, -0.9937886, 0.9927788

Time for backsubstitution: 20.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 927

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5664689, upper bound: 0.5646610
time: 5.01 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665197, upper bound: 0.5693666
time: 6.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.55 seconds
NS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 2, lower bound: -0.5664689, upper bound: 0.5646594
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 31.55
Output dim: 2, lower bound: -0.5665197, upper bound: 0.5693686
NS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 31.55
Output dim: 2, lower bound: -0.5664689, upper bound: 0.5646610
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 31.55
Output dim: 2, lower bound: -0.5665197, upper bound: 0.5693666

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -10.5007992, -8.6151867, -10.4845104, -8.6201143, -1.3286505, 1.3026590
1: -3.0011852, -1.2013631, -2.9913921, -1.2639771, -1.4892454, 1.5414186
2: 1.9156787, 3.3922994, 1.9922134, 3.3791318, -1.2584338, 1.2454863
3: -6.9948201, -5.5135183, -6.9479055, -5.5200844, -1.0384700, 0.9701185
4: -2.0801392, -0.6623383, -2.0611091, -0.6649580, -0.9589236, 0.9472477
5: -4.3501863, -2.9686570, -4.3437805, -3.0172141, -1.0663521, 1.0849125
6: -4.3488178, -2.4950233, -4.3318191, -2.5348859, -1.4197526, 1.4217820
7: -8.5299644, -7.1897502, -8.5212164, -7.2032061, -0.8465726, 0.8638296
8: -4.3281202, -2.6837451, -4.3130341, -2.7289662, -1.3893251, 1.4508920
9: -11.8632212, -10.1493120, -11.8572044, -10.1522942, -0.9824185, 0.9800756

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5682823
time: 4.37 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665181, upper bound: 0.5693710
time: 4.58 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -10.5030994, -8.6151752, -10.5030994, -8.6151810, -1.3420963, 1.3285990
1: -3.0012183, -1.2005291, -3.0012169, -1.2005281, -1.5190287, 1.5175419
2: 1.9155631, 3.3925023, 1.9155654, 3.3925028, -1.2799499, 1.2697730
3: -6.9952579, -5.5132847, -6.9952450, -5.5132847, -1.0312691, 0.9938395
4: -2.0814836, -0.6623101, -2.0814826, -0.6623135, -0.9587847, 0.9634240
5: -4.3505344, -2.9685149, -4.3505278, -2.9685142, -1.0925801, 1.0787759
6: -4.3493881, -2.4950233, -4.3493876, -2.4950278, -1.4382315, 1.4490156
7: -8.5299635, -7.1885796, -8.5299625, -7.1885858, -0.8695381, 0.8852111
8: -4.3286328, -2.6837451, -4.3286319, -2.6837535, -1.4292579, 1.4606915
9: -11.8640461, -10.1492834, -11.8640461, -10.1492834, -0.9925568, 0.9923606

Time for backsubstitution: 20.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5682808
time: 5.60 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665181, upper bound: 0.5693675
time: 4.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.86 seconds
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.86
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5682823
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.86
Output dim: 2, lower bound: -0.5665181, upper bound: 0.5693710
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.86
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5682808
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.86
Output dim: 2, lower bound: -0.5665181, upper bound: 0.5693675

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -10.4969273, -8.6154404, -10.4838238, -8.6202240, -1.3248358, 1.3020282
1: -2.9999738, -1.2028608, -2.9908278, -1.2641034, -1.4879127, 1.5393038
2: 1.9214015, 3.3903861, 1.9929415, 3.3782024, -1.2514992, 1.2428474
3: -6.9921479, -5.5157952, -6.9468966, -5.5205841, -1.0354877, 0.9669383
4: -2.0784097, -0.6646800, -2.0602689, -0.6653094, -0.9568994, 0.9441082
5: -4.3464928, -2.9696445, -4.3432722, -3.0176511, -1.0620897, 1.0835571
6: -4.3475628, -2.4955492, -4.3315964, -2.5349753, -1.4183092, 1.4210105
7: -8.5272207, -7.1935997, -8.5198317, -7.2036266, -0.8435998, 0.8587887
8: -4.3255172, -2.6849213, -4.3126173, -2.7295649, -1.3862395, 1.4493606
9: -11.8572111, -10.1561089, -11.8541431, -10.1535006, -0.9761775, 0.9711857

Time for backsubstitution: 20.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618087, upper bound: 0.5682186
time: 4.61 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5636331
time: 7.34 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -10.5007973, -8.6151876, -10.4845095, -8.6201153, -1.3286495, 1.3026373
1: -3.0011840, -1.2013633, -2.9913902, -1.2639778, -1.4881473, 1.5414171
2: 1.9156804, 3.3922977, 1.9922136, 3.3791323, -1.2557387, 1.2448282
3: -6.9948201, -5.5135193, -6.9479046, -5.5200834, -1.0378156, 0.9702590
4: -2.0801373, -0.6623392, -2.0611091, -0.6649590, -0.9586294, 0.9472460
5: -4.3501854, -2.9686575, -4.3437805, -3.0172141, -1.0665069, 1.0839791
6: -4.3488183, -2.4950235, -4.3318195, -2.5348854, -1.4197054, 1.4215205
7: -8.5299606, -7.1897516, -8.5212154, -7.2032061, -0.8444536, 0.8638282
8: -4.3281207, -2.6837444, -4.3130341, -2.7289667, -1.3893232, 1.4498756
9: -11.8632154, -10.1493139, -11.8572025, -10.1522961, -0.9801426, 0.9800727

Time for backsubstitution: 20.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618147, upper bound: 0.5693110
time: 4.32 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5618134, upper bound: 0.5646583
time: 4.83 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -10.4992237, -8.6154299, -10.5024118, -8.6152897, -1.3382797, 1.3279629
1: -3.0000072, -1.2020264, -3.0006514, -1.2006538, -1.5176954, 1.5154266
2: 1.9212878, 3.3905911, 1.9163002, 3.3915725, -1.2730174, 1.2671287
3: -6.9925861, -5.5155621, -6.9942451, -5.5137844, -1.0282483, 0.9906812
4: -2.0797544, -0.6646514, -2.0806446, -0.6626678, -0.9567574, 0.9602848
5: -4.3468404, -2.9695020, -4.3500090, -2.9689481, -1.0883226, 1.0773652
6: -4.3481307, -2.4955478, -4.3491507, -2.4951181, -1.4367876, 1.4482360
7: -8.5272217, -7.1924305, -8.5285797, -7.1890111, -0.8665617, 0.8801694
8: -4.3260279, -2.6849196, -4.3282070, -2.6843500, -1.4261703, 1.4591560
9: -11.8580379, -10.1560793, -11.8609867, -10.1504917, -0.9863396, 0.9822550

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618087, upper bound: 0.5682150
time: 4.39 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5618087, upper bound: 0.5636296
time: 4.30 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -10.5030985, -8.6151762, -10.5031013, -8.6151810, -1.3420949, 1.3285775
1: -3.0012169, -1.2005289, -3.0012159, -1.2005298, -1.5179319, 1.5175409
2: 1.9155641, 3.3924999, 1.9155655, 3.3925018, -1.2772512, 1.2676935
3: -6.9952555, -5.5132866, -6.9952435, -5.5132856, -1.0312290, 0.9939806
4: -2.0814815, -0.6623106, -2.0814829, -0.6623139, -0.9584889, 0.9634230
5: -4.3505330, -2.9685159, -4.3505273, -2.9685144, -1.0927343, 1.0787339
6: -4.3493876, -2.4950242, -4.3493867, -2.4950280, -1.4379983, 1.4487329
7: -8.5299616, -7.1885786, -8.5299625, -7.1885848, -0.8674192, 0.8844056
8: -4.3286314, -2.6837447, -4.3286324, -2.6837528, -1.4292560, 1.4602690
9: -11.8640404, -10.1492863, -11.8640451, -10.1492863, -0.9887910, 0.9887360

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618147, upper bound: 0.5693078
time: 4.42 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5618147, upper bound: 0.5646560
time: 4.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 31.00 seconds
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618087, upper bound: 0.5682186
NS_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5636331
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618147, upper bound: 0.5693110
NS_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618134, upper bound: 0.5646583
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618087, upper bound: 0.5682150
NS_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618087, upper bound: 0.5636296
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618147, upper bound: 0.5693078
NS_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 31.00
Output dim: 2, lower bound: -0.5618147, upper bound: 0.5646560

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -10.4933882, -8.6271591, -10.4838238, -8.6202240, -1.3199100, 1.2994199
1: -2.9962859, -1.2037241, -2.9908278, -1.2641034, -1.4843907, 1.5400066
2: 1.9319453, 3.3885403, 1.9929415, 3.3782024, -1.2411265, 1.2428215
3: -6.9529104, -5.5245848, -6.9468966, -5.5205841, -0.9962685, 0.9955592
4: -2.0771723, -0.6708250, -2.0602689, -0.6653094, -0.9564104, 0.9379022
5: -4.3287706, -2.9734020, -4.3432722, -3.0176511, -1.0441825, 1.0805357
6: -4.3448677, -2.5044518, -4.3315964, -2.5349753, -1.4233632, 1.4120345
7: -8.5235634, -7.2104692, -8.5198317, -7.2036266, -0.8555539, 0.8418834
8: -4.3171210, -2.7184925, -4.3126173, -2.7295649, -1.4094467, 1.4157760
9: -11.8567467, -10.1589622, -11.8541431, -10.1535006, -0.9769239, 0.9681935

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5568299, upper bound: 0.5675308
time: 4.39 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618052, upper bound: 0.5682167
time: 5.50 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -10.4972448, -8.6269073, -10.4845095, -8.6201153, -1.3237185, 1.3000293
1: -2.9974964, -1.2022305, -2.9913902, -1.2639778, -1.4846258, 1.5421152
2: 1.9262317, 3.3904505, 1.9922136, 3.3791323, -1.2453675, 1.2447689
3: -6.9555879, -5.5223079, -6.9479046, -5.5200834, -0.9986022, 0.9988792
4: -2.0789013, -0.6684895, -2.0611091, -0.6649590, -0.9576104, 0.9410373
5: -4.3324542, -2.9724154, -4.3437805, -3.0172141, -1.0485950, 1.0809639
6: -4.3461189, -2.5039272, -4.3318195, -2.5348854, -1.4247546, 1.4125440
7: -8.5263042, -7.2066226, -8.5212154, -7.2032061, -0.8564067, 0.8469200
8: -4.3197193, -2.7173181, -4.3130341, -2.7289667, -1.4125242, 1.4162910
9: -11.8627491, -10.1521759, -11.8572025, -10.1522961, -0.9808893, 0.9770789

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of NS_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5611276, upper bound: 0.5643082
time: 4.69 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618114, upper bound: 0.5693083
time: 5.59 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -10.4956884, -8.6271477, -10.5024118, -8.6152897, -1.3334017, 1.3253546
1: -2.9963186, -1.2028906, -3.0006514, -1.2006538, -1.5141544, 1.5161290
2: 1.9318290, 3.3887446, 1.9163002, 3.3915725, -1.2626433, 1.2647400
3: -6.9533482, -5.5243516, -6.9942451, -5.5137844, -0.9890532, 1.0142453
4: -2.0785177, -0.6707973, -2.0806446, -0.6626678, -0.9601552, 0.9540731
5: -4.3291168, -2.9732594, -4.3500090, -2.9689481, -1.0704155, 1.0874302
6: -4.3454366, -2.5044515, -4.3491507, -2.4951181, -1.4344683, 1.4392595
7: -8.5235643, -7.2092986, -8.5285797, -7.1890111, -0.8693235, 0.8632642
8: -4.3176336, -2.7184920, -4.3282070, -2.6843500, -1.4300013, 1.4256253
9: -11.8575745, -10.1589355, -11.8609867, -10.1504917, -0.9852228, 0.9792221

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5568299, upper bound: 0.5675274
time: 3.96 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618052, upper bound: 0.5682160
time: 4.80 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -10.4995480, -8.6268950, -10.5031013, -8.6151810, -1.3372116, 1.3259685
1: -2.9975288, -1.2013950, -3.0012159, -1.2005298, -1.5143909, 1.5182405
2: 1.9261154, 3.3906536, 1.9155655, 3.3925018, -1.2668796, 1.2653127
3: -6.9560232, -5.5220752, -6.9952435, -5.5132856, -0.9920411, 1.0166799
4: -2.0802450, -0.6684594, -2.0814829, -0.6623139, -0.9618874, 0.9572086
5: -4.3328018, -2.9722726, -4.3505273, -2.9685144, -1.0748224, 1.0887992
6: -4.3466887, -2.5039272, -4.3493867, -2.4950280, -1.4356794, 1.4397559
7: -8.5263042, -7.2054524, -8.5299625, -7.1885848, -0.8695171, 0.8674695
8: -4.3202314, -2.7173171, -4.3286324, -2.6837528, -1.4321179, 1.4267387
9: -11.8635788, -10.1521444, -11.8640451, -10.1492863, -0.9876776, 0.9857057

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5568421, upper bound: 0.5686118
time: 4.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618111, upper bound: 0.5693048
time: 5.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 32.45 seconds
NS_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5568299, upper bound: 0.5675308
NS_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5618052, upper bound: 0.5682167
NS_A2_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5611276, upper bound: 0.5643082
NS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5618114, upper bound: 0.5693083
NS_A2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5568299, upper bound: 0.5675274
NS_A2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5618052, upper bound: 0.5682160
NS_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5568421, upper bound: 0.5686118
NS_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 32.45
Output dim: 2, lower bound: -0.5618111, upper bound: 0.5693048

## BFS NS instance: NS_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -10.4925413, -8.6296740, -10.4834557, -8.6218748, -1.3139052, 1.2981205
1: -2.9672136, -1.2124624, -2.9782348, -1.2648473, -1.4649520, 1.5091767
2: 1.9372600, 3.3847227, 1.9950916, 3.3768713, -1.2342598, 1.2354763
3: -6.9519954, -5.5231895, -6.9464521, -5.5215764, -0.9909101, 0.9969230
4: -2.0798559, -0.6720123, -2.0599751, -0.6658072, -0.9486215, 0.9301472
5: -4.3261600, -2.9824479, -4.3427019, -3.0213716, -1.0464685, 1.0690844
6: -4.3388052, -2.5241942, -4.3307190, -2.5431566, -1.4061499, 1.3909249
7: -8.5145168, -7.2142601, -8.5160675, -7.2042446, -0.8465924, 0.8424000
8: -4.3102379, -2.7204609, -4.3097858, -2.7301192, -1.4006824, 1.4080300
9: -11.8573952, -10.1610117, -11.8540401, -10.1543283, -0.9741516, 0.9628947

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 143

## Relational analysis of NS_A2_B1_B2_A1_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5563275, upper bound: 0.5673886
time: 4.17 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5563009, upper bound: 0.5673874
time: 4.36 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -10.4925623, -8.6271982, -10.4837532, -8.6202269, -1.3177085, 1.2931931
1: -2.9961336, -1.2037258, -2.9908135, -1.2641029, -1.4685311, 1.5359030
2: 1.9319462, 3.3884706, 1.9929422, 3.3781962, -1.2391763, 1.2423167
3: -6.9521732, -5.5245852, -6.9468293, -5.5205836, -0.9949393, 0.9954922
4: -2.0771732, -0.6718106, -2.0602691, -0.6653905, -0.9554930, 0.9321344
5: -4.3287592, -2.9734385, -4.3432713, -3.0176542, -1.0434475, 1.0747902
6: -4.3447666, -2.5044556, -4.3315878, -2.5349767, -1.4232566, 1.3988338
7: -8.5235615, -7.2104897, -8.5198317, -7.2036285, -0.8508549, 0.8414090
8: -4.3169570, -2.7185049, -4.3126030, -2.7295687, -1.4081335, 1.4148710
9: -11.8567457, -10.1594830, -11.8541431, -10.1535454, -0.9767275, 0.9657590

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 143

## Relational analysis of NS_A2_B1_B2_A1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5612681, upper bound: 0.5680515
time: 4.79 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5617995, upper bound: 0.5682138
time: 7.61 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -10.4971752, -8.6269093, -10.4836626, -8.6201572, -1.3174934, 1.2978077
1: -2.9974825, -1.2022300, -2.9912257, -1.2639771, -1.4846139, 1.5262451
2: 1.9262323, 3.3904448, 1.9922147, 3.3790579, -1.2440779, 1.2392921
3: -6.9555254, -5.5223083, -6.9471111, -5.5200863, -0.9977207, 0.9993167
4: -2.0789015, -0.6685710, -2.0611088, -0.6659470, -0.9540691, 0.9408996
5: -4.3324528, -2.9724185, -4.3437700, -3.0172501, -1.0587809, 1.0786564
6: -4.3461099, -2.5039263, -4.3317122, -2.5348911, -1.4116495, 1.4086888
7: -8.5263042, -7.2066250, -8.5212116, -7.2032270, -0.8554711, 0.8542438
8: -4.3197041, -2.7173195, -4.3128524, -2.7289798, -1.4125061, 1.4152822
9: -11.8627520, -10.1522160, -11.8572006, -10.1528225, -0.9784555, 0.9768839

Time for backsubstitution: 22.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 143
type: A, layer: 1, pos: 143
type: B, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 143

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5616485, upper bound: 0.5687750
time: 5.30 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618061, upper bound: 0.5687542
time: 7.34 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.15 + 550.33 = 607.48 seconds
