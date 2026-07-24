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
execution time: IAR + RelationalAnalysis = 22.27 + 35.60 = 57.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.5694884, upper bound: 0.5694897

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693774, upper bound: 0.5665206
time: 5.41 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693739, upper bound: 0.5693724
time: 4.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.42 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.42
Output dim: 2, lower bound: -0.5693774, upper bound: 0.5665206
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.42
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

Time for backsubstitution: 20.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5665213, upper bound: 0.5665210
time: 6.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5665199
time: 6.83 seconds

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

Time for backsubstitution: 20.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693720
time: 5.05 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693713
time: 6.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.98 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.98
Output dim: 2, lower bound: -0.5665213, upper bound: 0.5665210
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 31.98
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5665199
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.98
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693720
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.98
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

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618151, upper bound: 0.5693086
time: 4.27 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665182, upper bound: 0.5693668
time: 4.62 seconds

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

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 927

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618151, upper bound: 0.5693107
time: 4.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665182, upper bound: 0.5693689
time: 6.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.46 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.46
Output dim: 2, lower bound: -0.5618151, upper bound: 0.5693086
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.46
Output dim: 2, lower bound: -0.5665182, upper bound: 0.5693668
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.46
Output dim: 2, lower bound: -0.5618151, upper bound: 0.5693107
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.46
Output dim: 2, lower bound: -0.5665182, upper bound: 0.5693689

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.4972458, -8.6269054, -10.4840965, -8.6250076, -1.3216085, 1.2993860
1: -2.9974971, -1.2022295, -2.9899566, -1.2641191, -1.4855819, 1.5407104
2: 1.9262303, 3.3904529, 1.9965570, 3.3790271, -1.2498932, 1.2411470
3: -6.9555893, -5.5223074, -6.9315968, -5.5201349, -1.0079055, 0.9824007
4: -2.0789020, -0.6684885, -2.0610929, -0.6674805, -0.9582615, 0.9410274
5: -4.3324552, -2.9724145, -4.3364921, -3.0173194, -1.0483556, 1.0813746
6: -4.3461185, -2.5039272, -4.3315086, -2.5386014, -1.4208887, 1.4146118
7: -8.5263062, -7.2066231, -8.5211678, -7.2102304, -0.8514826, 0.8468535
8: -4.3197193, -2.7173166, -4.3126578, -2.7429633, -1.3985343, 1.4233160
9: -11.8627567, -10.1521740, -11.8571939, -10.1534414, -0.9819381, 0.9770677

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5693050
time: 5.86 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618132, upper bound: 0.5693109
time: 5.20 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.5007982, -8.6151915, -10.4845095, -8.6201115, -1.3151169, 1.3161559
1: -3.0011835, -1.2013626, -2.9913936, -1.2639778, -1.4877582, 1.5429049
2: 1.9156808, 3.3922987, 1.9922105, 3.3791318, -1.2536206, 1.2392125
3: -6.9948072, -5.5135174, -6.9479184, -5.5200844, -1.0096269, 1.0075479
4: -2.0801394, -0.6623406, -2.0611095, -0.6649551, -0.9629922, 0.9426070
5: -4.3501787, -2.9686573, -4.3437872, -3.0172141, -1.0525455, 1.0952783
6: -4.3488188, -2.4950271, -4.3318200, -2.5348806, -1.4240358, 1.4162166
7: -8.5299644, -7.1897583, -8.5212164, -7.2031975, -0.8622453, 0.8481566
8: -4.3281198, -2.6837523, -4.3130341, -2.7289610, -1.4207592, 1.4256876
9: -11.8632231, -10.1493130, -11.8572025, -10.1522942, -0.9836342, 0.9788604

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693686
time: 4.42 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5693687
time: 4.73 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.4995499, -8.6268950, -10.5026855, -8.6200724, -1.3350649, 1.3252811
1: -2.9975314, -1.2013953, -2.9997835, -1.2006700, -1.5153484, 1.5168200
2: 1.9261140, 3.3906567, 1.9199132, 3.3924007, -1.2714117, 1.2667546
3: -6.9560251, -5.5220747, -6.9789267, -5.5133328, -0.9920321, 1.0061166
4: -2.0802472, -0.6684589, -2.0814674, -0.6648350, -0.9596263, 0.9571984
5: -4.3328028, -2.9722724, -4.3432407, -2.9686189, -1.0745826, 1.0814273
6: -4.3466902, -2.5039265, -4.3490896, -2.4987435, -1.4358330, 1.4404812
7: -8.5263062, -7.2054505, -8.5299158, -7.1956148, -0.8720303, 0.8682351
8: -4.3202324, -2.7173171, -4.3282762, -2.6977484, -1.4332910, 1.4268820
9: -11.8635845, -10.1521435, -11.8640375, -10.1504326, -0.9911826, 0.9897314

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5693015
time: 5.26 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618132, upper bound: 0.5693111
time: 4.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.5030994, -8.6151810, -10.5030994, -8.6151752, -1.3285990, 1.3420961
1: -3.0012169, -1.2005281, -3.0012183, -1.2005291, -1.5175419, 1.5190287
2: 1.9155654, 3.3925028, 1.9155631, 3.3925023, -1.2751474, 1.2745757
3: -6.9952450, -5.5132847, -6.9952579, -5.5132847, -0.9938395, 1.0312693
4: -2.0814826, -0.6623135, -2.0814836, -0.6623101, -0.9634243, 0.9587847
5: -4.3505278, -2.9685142, -4.3505344, -2.9685149, -1.0787759, 1.0925798
6: -4.3493876, -2.4950278, -4.3493881, -2.4950233, -1.4437609, 1.4434862
7: -8.5299625, -7.1885858, -8.5299635, -7.1885796, -0.8852110, 0.8695381
8: -4.3286319, -2.6837535, -4.3286328, -2.6837451, -1.4606915, 1.4292579
9: -11.8640461, -10.1492834, -11.8640461, -10.1492834, -0.9933705, 0.9915469

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4632
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 120

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693666
time: 4.59 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5693662
time: 4.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.85 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5693050
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5618132, upper bound: 0.5693109
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693686
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5693687
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5693015
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5618132, upper bound: 0.5693111
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693666
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.85
Output dim: 2, lower bound: -0.5665164, upper bound: 0.5693662

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.4965677, -8.6270151, -10.4802666, -8.6252584, -1.3209872, 1.2955856
1: -2.9969313, -1.2023535, -2.9887447, -1.2656043, -1.4834871, 1.5393786
2: 1.9269664, 3.3895235, 2.0022559, 3.3771100, -1.2472367, 1.2346206
3: -6.9545884, -5.5228062, -6.9289012, -5.5224118, -1.0045238, 0.9793315
4: -2.0780635, -0.6688433, -2.0593584, -0.6698155, -0.9548042, 0.9389939
5: -4.3319349, -2.9728487, -4.3328214, -3.0183132, -1.0469408, 1.0770559
6: -4.3458838, -2.5040164, -4.3302717, -2.5391254, -1.4201155, 1.4131815
7: -8.5249205, -7.2070475, -8.5184250, -7.2140608, -0.8464737, 0.8438773
8: -4.3193002, -2.7179165, -4.3100753, -2.7441404, -1.3970027, 1.4202540
9: -11.8596954, -10.1533813, -11.8511858, -10.1602240, -0.9730523, 0.9708300

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5682189
time: 6.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5607213, upper bound: 0.5693038
time: 4.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -10.4972458, -8.6269073, -10.4840946, -8.6250057, -1.3215871, 1.2993855
1: -2.9974961, -1.2022295, -2.9899552, -1.2641194, -1.4855809, 1.5396066
2: 1.9262303, 3.3904517, 1.9965591, 3.3790259, -1.2478209, 1.2411442
3: -6.9555883, -5.5223083, -6.9315944, -5.5201349, -1.0069609, 0.9823611
4: -2.0789018, -0.6684885, -2.0610909, -0.6674819, -0.9572237, 0.9407320
5: -4.3324537, -2.9724147, -4.3364916, -3.0173202, -1.0483141, 1.0799839
6: -4.3461189, -2.5039268, -4.3315077, -2.5386014, -1.4208536, 1.4144158
7: -8.5263042, -7.2066226, -8.5211639, -7.2102308, -0.8514814, 0.8447341
8: -4.3197188, -2.7173171, -4.3126569, -2.7429652, -1.3981123, 1.4232349
9: -11.8627529, -10.1521740, -11.8571882, -10.1534414, -0.9819350, 0.9747922

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682191
time: 5.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682204
time: 4.58 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.5001106, -8.6153002, -10.4806767, -8.6203651, -1.3144960, 1.3123591
1: -3.0006175, -1.2014899, -2.9901803, -1.2654648, -1.4856620, 1.5415688
2: 1.9164158, 3.3913684, 1.9979122, 3.3772154, -1.2509656, 1.2326815
3: -6.9938102, -5.5140171, -6.9452276, -5.5223627, -1.0062506, 1.0044818
4: -2.0793014, -0.6626959, -2.0593753, -0.6672902, -0.9595335, 0.9405737
5: -4.3496604, -2.9690909, -4.3401127, -3.0182078, -1.0511317, 1.0909591
6: -4.3485813, -2.4951196, -4.3305807, -2.5354052, -1.4232531, 1.4147832
7: -8.5285797, -7.1901836, -8.5184736, -7.2070303, -0.8572357, 0.8451810
8: -4.3276968, -2.6843503, -4.3104491, -2.7301359, -1.4192252, 1.4226246
9: -11.8601580, -10.1505203, -11.8511963, -10.1590786, -0.9747477, 0.9726224

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5682826
time: 4.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693686
time: 4.39 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.5007982, -8.6151915, -10.4845076, -8.6201124, -1.3150954, 1.3161550
1: -3.0011849, -1.2013619, -2.9913907, -1.2639773, -1.4877567, 1.5418029
2: 1.9156816, 3.3922985, 1.9922112, 3.3791304, -1.2515411, 1.2392108
3: -6.9948082, -5.5135193, -6.9479151, -5.5200849, -1.0086832, 1.0075078
4: -2.0801387, -0.6623421, -2.0611076, -0.6649561, -0.9619524, 0.9423110
5: -4.3501787, -2.9686570, -4.3437872, -3.0172155, -1.0525031, 1.0938842
6: -4.3488188, -2.4950271, -4.3318195, -2.5348809, -1.4239988, 1.4160190
7: -8.5299625, -7.1897583, -8.5212145, -7.2031999, -0.8622444, 0.8460374
8: -4.3281198, -2.6837533, -4.3130341, -2.7289615, -1.4203362, 1.4255311
9: -11.8632164, -10.1493130, -11.8571968, -10.1522961, -0.9836316, 0.9765842

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665161, upper bound: 0.5682839
time: 5.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665162, upper bound: 0.5682851
time: 4.24 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.4988708, -8.6270027, -10.4988174, -8.6203270, -1.3344278, 1.3214583
1: -2.9969640, -1.2015188, -2.9985726, -1.2021666, -1.5132327, 1.5154920
2: 1.9268506, 3.3897269, 1.9256335, 3.3904889, -1.2687676, 1.2598259
3: -6.9550233, -5.5225730, -6.9762526, -5.5156112, -0.9888697, 1.0030930
4: -2.0794084, -0.6688151, -2.0797384, -0.6671739, -0.9564896, 0.9551697
5: -4.3322830, -2.9727066, -4.3395510, -2.9696064, -1.0731719, 1.0771735
6: -4.3464541, -2.5040174, -4.3478374, -2.4992690, -1.4350553, 1.4390318
7: -8.5249205, -7.2058764, -8.5271730, -7.1994648, -0.8664682, 0.8652586
8: -4.3198118, -2.7179155, -4.3256745, -2.6989241, -1.4317589, 1.4237990
9: -11.8605232, -10.1533527, -11.8580265, -10.1572247, -0.9810772, 0.9835145

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5682155
time: 4.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5693015
time: 5.24 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.4995470, -8.6268950, -10.5026846, -8.6200743, -1.3350415, 1.3252797
1: -2.9975300, -1.2013965, -2.9997818, -1.2006683, -1.5153470, 1.5157213
2: 1.9261147, 3.3906546, 1.9199153, 3.3923993, -1.2693381, 1.2640612
3: -6.9560237, -5.5220752, -6.9789243, -5.5133343, -0.9921722, 1.0060768
4: -2.0802460, -0.6684599, -2.0814657, -0.6648350, -0.9596251, 0.9569018
5: -4.3328028, -2.9722722, -4.3432398, -2.9686198, -1.0745404, 1.0815828
6: -4.3466897, -2.5039268, -4.3490882, -2.4987447, -1.4355521, 1.4404335
7: -8.5263052, -7.2054501, -8.5299129, -7.1956148, -0.8698750, 0.8661166
8: -4.3202329, -2.7173171, -4.3282757, -2.6977484, -1.4322672, 1.4268801
9: -11.8635807, -10.1521435, -11.8640299, -10.1504345, -0.9875604, 0.9859691

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682169
time: 5.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682169
time: 4.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.5024118, -8.6152897, -10.4992237, -8.6154299, -1.3279629, 1.3382795
1: -3.0006514, -1.2006538, -3.0000072, -1.2020264, -1.5154266, 1.5176954
2: 1.9163002, 3.3915725, 1.9212878, 3.3905911, -1.2725034, 1.2676427
3: -6.9942451, -5.5137844, -6.9925861, -5.5155621, -0.9906812, 1.0282485
4: -2.0806446, -0.6626678, -2.0797544, -0.6646514, -0.9602847, 0.9567573
5: -4.3500090, -2.9689481, -4.3468404, -2.9695020, -1.0773649, 1.0883224
6: -4.3491507, -2.4951181, -4.3481307, -2.4955478, -1.4429812, 1.4420421
7: -8.5285797, -7.1890111, -8.5272217, -7.1924305, -0.8798454, 0.8665614
8: -4.3282070, -2.6843500, -4.3260279, -2.6849196, -1.4591556, 1.4261703
9: -11.8609867, -10.1504917, -11.8580379, -10.1560793, -0.9832647, 0.9853296

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5682791
time: 4.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693652
time: 4.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.5031013, -8.6151810, -10.5030985, -8.6151762, -1.3285775, 1.3420951
1: -3.0012159, -1.2005298, -3.0012169, -1.2005289, -1.5175409, 1.5179319
2: 1.9155655, 3.3925018, 1.9155641, 3.3924999, -1.2730680, 1.2718766
3: -6.9952435, -5.5132856, -6.9952555, -5.5132866, -0.9939809, 1.0312290
4: -2.0814829, -0.6623139, -2.0814815, -0.6623106, -0.9634228, 0.9584889
5: -4.3505273, -2.9685144, -4.3505330, -2.9685159, -1.0787337, 1.0927343
6: -4.3493867, -2.4950280, -4.3493876, -2.4950242, -1.4434781, 1.4432530
7: -8.5299625, -7.1885848, -8.5299616, -7.1885786, -0.8832514, 0.8674192
8: -4.3286324, -2.6837528, -4.3286314, -2.6837447, -1.4597178, 1.4292560
9: -11.8640451, -10.1492863, -11.8640404, -10.1492863, -0.9897459, 0.9877812

Time for backsubstitution: 22.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4632
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 120

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4632

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665161, upper bound: 0.5682831
time: 4.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665162, upper bound: 0.5682807
time: 4.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 32.01 seconds
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5682189
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5607213, upper bound: 0.5693038
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682191
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682204
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5682826
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693686
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5665161, upper bound: 0.5682839
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5665162, upper bound: 0.5682851
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5682155
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5607226, upper bound: 0.5693015
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682169
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5618074, upper bound: 0.5682169
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5682791
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5654300, upper bound: 0.5693652
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5665161, upper bound: 0.5682831
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 2, lower bound: -0.5665162, upper bound: 0.5682807

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.4933882, -8.6271591, -10.4802666, -8.6252584, -1.3176508, 1.2954359
1: -2.9962859, -1.2037241, -2.9887447, -1.2656043, -1.4829483, 1.5380578
2: 1.9319453, 3.3885403, 2.0022559, 3.3771100, -1.2422829, 1.2335842
3: -6.9529104, -5.5245848, -6.9289012, -5.5224118, -1.0030849, 0.9775009
4: -2.0771723, -0.6708250, -2.0593584, -0.6698155, -0.9542377, 0.9370022
5: -4.3287706, -2.9734020, -4.3328214, -3.0183132, -1.0435815, 1.0766411
6: -4.3448677, -2.5044518, -4.3302717, -2.5391254, -1.4190154, 1.4127438
7: -8.5235634, -7.2104692, -8.5184250, -7.2140608, -0.8451118, 0.8404536
8: -4.3171210, -2.7184925, -4.3100753, -2.7441404, -1.3948483, 1.4196479
9: -11.8567467, -10.1589622, -11.8511858, -10.1602240, -0.9701123, 0.9652388

Time for backsubstitution: 22.57 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.86 + 545.22 = 603.08 seconds
