## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.9085582323
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.4833460, 2.4833460)
1: (-19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.7038298, 3.7038298)
2: (-6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.8991394, 2.8991392)
3: (-10.8192272, -7.7928076, -10.8192272, -7.7928076, -3.0264196, 3.0264196)
4: (-13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.9983845, 2.9983845)
5: (-4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.4810138, 2.4810138)
6: (-4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.5990338, 2.5990338)
7: (-12.8235607, -8.7824364, -12.8235607, -8.7824364, -4.0411243, 4.0411243)
8: (-5.4501801, -3.1462440, -5.4501801, -3.1462440, -2.2276306, 2.2276306)
9: (-1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725)

## BASE Result
execution time: IAR + LP analysis = 14.77 + 33.19 = 47.96 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.04 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2897982597351074
rel_dist={0: [-1.319613808150022, 1.3196135105828688]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0339293479919434
rel_dist={0: [-0.9094671207244254, 0.9094665870331546]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8633503913879395
rel_dist={0: [-0.5847592780895585, 0.5847577424659303]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.9486398696899414
rel_dist={0: [-0.7535712826074459, 0.7535698485994864]}

## Binary Search Result
Binary search time: 210.31 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3341.73 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271530, upper bound: 1.4199076
time: 5.09 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297444, upper bound: 1.4297441
time: 8.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.35 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.35
Output dim: 0, lower bound: -1.4271530, upper bound: 1.4199076
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.35
Output dim: 0, lower bound: -1.4297444, upper bound: 1.4297441

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 7.7684388, 10.2174206, 7.7562246, 10.2317152, -2.3550057, 2.3527205
1: -19.2248173, -15.2928391, -19.2507172, -15.2747021, -2.9626026, 2.9696641
2: -6.5115714, -3.5520163, -6.5208359, -3.5497639, -2.3444347, 2.3522639
3: -10.7933292, -7.8039856, -10.8126593, -7.7936406, -2.7648416, 2.7679377
4: -13.5695906, -10.6057529, -13.5847692, -10.5941849, -2.6894341, 2.6934233
5: -4.6272192, -2.1693683, -4.6367040, -2.1612105, -2.0577912, 2.0580378
6: -4.4925184, -1.9654677, -4.5138092, -1.9300066, -2.3516178, 2.3376031
7: -12.8079023, -8.8138399, -12.8221092, -8.7906666, -3.4963875, 3.4902515
8: -5.4340787, -3.1679492, -5.4483004, -3.1522942, -1.7692018, 1.7681696
9: -1.8838596, 1.0316887, -1.9187107, 1.0461683, -2.9300280, 2.9503994

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4183916, upper bound: 1.4198880
time: 5.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271330, upper bound: 1.4198879
time: 5.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7540379, 10.2373838, -2.3750849, 2.3651075
1: -19.2597046, -15.2714100, -19.2597141, -15.2714062, -2.9893885, 3.0001583
2: -6.5238457, -3.5489109, -6.5238509, -3.5489068, -2.3601170, 2.3606622
3: -10.8192225, -7.7928071, -10.8192301, -7.7928081, -2.7902832, 2.7990522
4: -13.5904970, -10.5921268, -13.5905075, -10.5921230, -2.7003412, 2.7115874
5: -4.6404028, -2.1593966, -4.6404066, -2.1593926, -2.0707364, 2.0711203
6: -4.5149150, -1.9159002, -4.5149164, -1.9158841, -2.3887258, 2.3505454
7: -12.8235588, -8.7824440, -12.8235626, -8.7824364, -3.5195074, 3.5063138
8: -5.4501791, -3.1462469, -5.4501805, -3.1462440, -1.7919750, 1.7753747
9: -1.9316473, 1.0465150, -1.9316578, 1.0465157, -2.9659967, 2.9781728

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4199074, upper bound: 1.4271532
time: 5.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4199075, upper bound: 1.4297449
time: 6.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.33
Output dim: 0, lower bound: -1.4183916, upper bound: 1.4198880
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.33
Output dim: 0, lower bound: -1.4271330, upper bound: 1.4198879
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.33
Output dim: 0, lower bound: -1.4199074, upper bound: 1.4271532
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.33
Output dim: 0, lower bound: -1.4199075, upper bound: 1.4297449

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 7.7694950, 10.2115049, 7.7659798, 10.2055101, -2.3271675, 2.3368480
1: -19.2232952, -15.2934570, -19.2444057, -15.2787638, -2.9564505, 2.9646630
2: -6.5085816, -3.5523045, -6.5081263, -3.5531652, -2.3379011, 2.3302989
3: -10.7927866, -7.8048296, -10.8089476, -7.7984490, -2.7570901, 2.7612038
4: -13.5649776, -10.6061420, -13.5634708, -10.5992756, -2.6784515, 2.6719012
5: -4.6268711, -2.1711395, -4.6343374, -2.1692119, -2.0478830, 2.0532026
6: -4.4914341, -1.9687774, -4.5070734, -1.9446497, -2.3394489, 2.3258362
7: -12.8019123, -8.8142977, -12.7961931, -8.7967701, -3.4832120, 3.4637737
8: -5.4332557, -3.1706104, -5.4426084, -3.1645365, -1.7562008, 1.7545247
9: -1.8778095, 1.0316250, -1.8916035, 1.0419250, -2.9197345, 2.9232285

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4183915, upper bound: 1.4111448
time: 5.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4183917, upper bound: 1.4198880
time: 8.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 7.7684388, 10.2174206, 7.7562261, 10.2317104, -2.3367853, 2.3527186
1: -19.2248173, -15.2928391, -19.2507133, -15.2747049, -2.9624567, 2.9669905
2: -6.5115714, -3.5520163, -6.5208335, -3.5497642, -2.3444347, 2.3469331
3: -10.7933292, -7.8039856, -10.8126583, -7.7936420, -2.7763224, 2.7661476
4: -13.5695906, -10.6057529, -13.5847673, -10.5941868, -2.6888666, 2.6840129
5: -4.6272192, -2.1693683, -4.6367035, -2.1612148, -2.0571914, 2.0618122
6: -4.4925184, -1.9654677, -4.5138092, -1.9300094, -2.3449550, 2.3369303
7: -12.8079023, -8.8138399, -12.8221054, -8.7906647, -3.4961538, 3.4746547
8: -5.4340787, -3.1679492, -5.4482989, -3.1522942, -1.7612860, 1.7681684
9: -1.8838596, 1.0316887, -1.9187059, 1.0461695, -2.9300292, 2.9503946

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271333, upper bound: 1.4111447
time: 5.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271332, upper bound: 1.4198880
time: 5.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7684388, 10.2174206, -2.3550329, 2.3604584
1: -19.2597046, -15.2714100, -19.2248173, -15.2928391, -2.9785867, 2.9658442
2: -6.5238457, -3.5489109, -6.5115714, -3.5520163, -2.3562489, 2.3458333
3: -10.8192225, -7.7928071, -10.7933292, -7.8039856, -2.7762485, 2.7630272
4: -13.5904970, -10.5921268, -13.5695906, -10.6057529, -2.6989813, 2.6907439
5: -4.6404028, -2.1593966, -4.6272192, -2.1693683, -2.0619636, 2.0580239
6: -4.5149150, -1.9159002, -4.4925184, -1.9654677, -2.3389831, 2.3658128
7: -12.8235588, -8.7824440, -12.8079023, -8.8138399, -3.4890523, 3.5059524
8: -5.4501791, -3.1462469, -5.4340787, -3.1679492, -1.7703276, 1.7752531
9: -1.9316473, 1.0465150, -1.8838596, 1.0316887, -2.9633360, 2.9303746

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198877, upper bound: 1.4183917
time: 5.43 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4271331
time: 5.05 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7540426, 10.2373781, -2.3651042, 2.3651042
1: -19.2597046, -15.2714100, -19.2597046, -15.2714100, -2.9893837, 2.9893837
2: -6.5238457, -3.5489109, -6.5238457, -3.5489109, -2.3606558, 2.3606558
3: -10.8192225, -7.7928071, -10.8192225, -7.7928071, -2.7990422, 2.7990422
4: -13.5904970, -10.5921268, -13.5904970, -10.5921268, -2.7003374, 2.7003374
5: -4.6404028, -2.1593966, -4.6404028, -2.1593966, -2.0707316, 2.0707319
6: -4.5149150, -1.9159002, -4.5149150, -1.9159002, -2.3505440, 2.3505440
7: -12.8235588, -8.7824440, -12.8235588, -8.7824440, -3.5063095, 3.5063100
8: -5.4501791, -3.1462469, -5.4501791, -3.1462469, -1.7753718, 1.7753716
9: -1.9316473, 1.0465150, -1.9316473, 1.0465150, -2.9659948, 2.9659948

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4209822
time: 5.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4297243
time: 5.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.50 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4183915, upper bound: 1.4111448
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4183917, upper bound: 1.4198880
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4271333, upper bound: 1.4111447
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4271332, upper bound: 1.4198880
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4198877, upper bound: 1.4183917
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4271331
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4209822
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.50
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4297243

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7659798, 10.2055101, -2.3182335, 2.3160264
1: -19.2184887, -15.2969704, -19.2444057, -15.2787638, -2.9541178, 2.9611435
2: -6.4989195, -3.5554147, -6.5081263, -3.5531652, -2.3192968, 2.3270717
3: -10.7896709, -7.8088303, -10.8089476, -7.7984490, -2.7524686, 2.7554746
4: -13.5483303, -10.6108570, -13.5634708, -10.5992756, -2.6628599, 2.6668482
5: -4.6248550, -2.1774464, -4.6343374, -2.1692119, -2.0451412, 2.0452917
6: -4.4857645, -1.9801078, -4.5070734, -1.9446497, -2.3328104, 2.3188119
7: -12.7819271, -8.8199348, -12.7961931, -8.7967701, -3.4638038, 3.4576855
8: -5.4283876, -3.1802011, -5.4426084, -3.1645365, -1.7461560, 1.7451668
9: -1.8567619, 1.0274441, -1.8916035, 1.0419250, -2.8986869, 2.9190476

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111468
time: 5.24 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111471
time: 5.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7659798, 10.2055101, -2.3280110, 2.3428082
1: -19.2248135, -15.2929382, -19.2444057, -15.2787638, -2.9582634, 2.9651837
2: -6.5115609, -3.5520177, -6.5081263, -3.5531652, -2.3408651, 2.3306193
3: -10.7929115, -7.8039832, -10.8089476, -7.7984490, -2.7558732, 2.7601709
4: -13.5695877, -10.6058292, -13.5634708, -10.5992756, -2.6839809, 2.6716051
5: -4.6272163, -2.1694930, -4.6343374, -2.1692119, -2.0474682, 2.0543358
6: -4.4925151, -1.9654703, -4.5070734, -1.9446497, -2.3400288, 2.3296175
7: -12.8078575, -8.8138428, -12.7961931, -8.7967701, -3.4893212, 3.4644260
8: -5.4340730, -3.1679649, -5.4426084, -3.1645365, -1.7571464, 1.7572029
9: -1.8838558, 1.0316219, -1.8916035, 1.0419250, -2.9257808, 2.9232254

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198882
time: 5.93 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198883
time: 5.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7564430, 10.2317085, -2.3420677, 2.3257236
1: -19.2184887, -15.2969704, -19.2507153, -15.2747984, -2.9580994, 2.9652710
2: -6.4989195, -3.5554147, -6.5208244, -3.5497661, -2.3228493, 2.3486981
3: -10.7896709, -7.8088303, -10.8122406, -7.7936420, -2.7571330, 2.7589445
4: -13.5483303, -10.6108570, -13.5847673, -10.5942631, -2.6676645, 2.6880188
5: -4.6248550, -2.1774464, -4.6367025, -2.1613379, -2.0540915, 2.0476270
6: -4.4857645, -1.9801078, -4.5138073, -1.9300106, -2.3436184, 2.3260050
7: -12.7819271, -8.8199348, -12.8220654, -8.7906647, -3.4705625, 3.4832096
8: -5.4283876, -3.1802011, -5.4482994, -3.1523080, -1.7581882, 1.7561104
9: -1.8567619, 1.0274441, -1.9187059, 1.0461011, -2.9028630, 2.9461501

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111444
time: 5.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111448
time: 6.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7562261, 10.2317104, -2.3367839, 2.3344989
1: -19.2248154, -15.2928429, -19.2507133, -15.2747049, -2.9599466, 2.9669886
2: -6.5115700, -3.5520153, -6.5208335, -3.5497642, -2.3391027, 2.3469331
3: -10.7933302, -7.8039842, -10.8126583, -7.7936420, -2.7763195, 2.7793427
4: -13.5695877, -10.6057549, -13.5847673, -10.5941868, -2.6800599, 2.6840110
5: -4.6272178, -2.1693721, -4.6367035, -2.1612148, -2.0616026, 2.0618100
6: -4.4925199, -1.9654692, -4.5138092, -1.9300094, -2.3449516, 2.3309331
7: -12.8078947, -8.8138428, -12.8221054, -8.7906647, -3.4807715, 3.4746532
8: -5.4340749, -3.1679506, -5.4482989, -3.1522942, -1.7612841, 1.7602530
9: -1.8838563, 1.0316887, -1.9187059, 1.0461695, -2.9300258, 2.9503946

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198880
time: 5.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198880
time: 6.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7694950, 10.2115049, -2.3391843, 2.3326216
1: -19.2533894, -15.2754498, -19.2232952, -15.2934570, -2.9735851, 2.9597111
2: -6.5111165, -3.5523076, -6.5085816, -3.5523045, -2.3342638, 2.3393028
3: -10.8155174, -7.7976027, -10.7927866, -7.8048296, -2.7695169, 2.7552867
4: -13.5691833, -10.5972061, -13.5649776, -10.6061420, -2.6774445, 2.6797733
5: -4.6380377, -2.1673660, -4.6268711, -2.1711395, -2.0571337, 2.0481665
6: -4.5081902, -1.9305443, -4.4914341, -1.9687774, -2.3272238, 2.3536401
7: -12.7976437, -8.7885475, -12.8019123, -8.8142977, -3.4625707, 3.4927869
8: -5.4444995, -3.1584921, -5.4332557, -3.1706104, -1.7566984, 1.7622514
9: -1.9045305, 1.0422702, -1.8778095, 1.0316250, -2.9361556, 2.9200797

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4183917
time: 5.98 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4183916
time: 5.61 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 7.7540431, 10.2373714, 7.7684388, 10.2174206, -2.3550310, 2.3422382
1: -19.2597008, -15.2714119, -19.2248173, -15.2928391, -2.9759140, 2.9656940
2: -6.5238419, -3.5489097, -6.5115714, -3.5520163, -2.3509188, 2.3458333
3: -10.8192215, -7.7928095, -10.7933292, -7.8039856, -2.7744555, 2.7745080
4: -13.5904942, -10.5921278, -13.5695906, -10.6057529, -2.6895561, 2.6901755
5: -4.6404009, -2.1593976, -4.6272192, -2.1693683, -2.0657387, 2.0574188
6: -4.5149145, -1.9159026, -4.4925184, -1.9654677, -2.3383079, 2.3591428
7: -12.8235540, -8.7824469, -12.8079023, -8.8138399, -3.4734492, 3.5057168
8: -5.4501781, -3.1462479, -5.4340787, -3.1679492, -1.7703266, 1.7673364
9: -1.9316421, 1.0465150, -1.8838596, 1.0316887, -2.9633307, 2.9303746

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4271332
time: 5.77 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111444, upper bound: 1.4271333
time: 5.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7550731, 10.2314625, -2.3491373, 2.3372629
1: -19.2533894, -15.2754498, -19.2581863, -15.2720079, -2.9843864, 2.9832339
2: -6.5111165, -3.5523076, -6.5208402, -3.5491958, -2.3386598, 2.3541069
3: -10.8155174, -7.7976027, -10.8186674, -7.7936425, -2.7923136, 2.7912593
4: -13.5691833, -10.5972061, -13.5858727, -10.5925064, -2.6787863, 2.6893353
5: -4.6380377, -2.1673660, -4.6400533, -2.1611423, -2.0658946, 2.0607667
6: -4.5081902, -1.9305443, -4.5138359, -1.9192102, -2.3387804, 2.3383689
7: -12.7976437, -8.7885475, -12.8175869, -8.7829056, -3.4799032, 3.4931822
8: -5.4444995, -3.1584921, -5.4493790, -3.1489077, -1.7617376, 1.7623818
9: -1.9045305, 1.0422702, -1.9255891, 1.0464511, -2.9386234, 2.9565268

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4137373, upper bound: 1.4209823
time: 5.85 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4209836
time: 7.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7540431, 10.2373714, 7.7540426, 10.2373781, -2.3651018, 2.3468845
1: -19.2597008, -15.2714119, -19.2597046, -15.2714100, -2.9867616, 2.9892340
2: -6.5238419, -3.5489097, -6.5238457, -3.5489109, -2.3553257, 2.3606558
3: -10.8192215, -7.7928095, -10.8192225, -7.7928071, -2.7972493, 2.8104548
4: -13.5904942, -10.5921278, -13.5904970, -10.5921268, -2.6909199, 2.6997685
5: -4.6404009, -2.1593976, -4.6404028, -2.1593966, -2.0745697, 2.0701268
6: -4.5149145, -1.9159026, -4.5149150, -1.9159002, -2.3498678, 2.3438997
7: -12.8235540, -8.7824469, -12.8235588, -8.7824440, -3.4907351, 3.5060759
8: -5.4501781, -3.1462479, -5.4501791, -3.1462469, -1.7753699, 1.7674553
9: -1.9316421, 1.0465150, -1.9316473, 1.0465150, -2.9625845, 2.9659948

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4297248
time: 5.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4297240
time: 8.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.96 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111468
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111471
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198882
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198883
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111444
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4111448
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198880
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198880
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4183917
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4183916
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4271332
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4111444, upper bound: 1.4271333
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4137373, upper bound: 1.4209823
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4209836
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4297248
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4297240

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7782664, 10.1912088, -2.3036389, 2.3036389
1: -19.2184887, -15.2969704, -19.2184887, -15.2969704, -2.9357505, 2.9357514
2: -6.4989195, -3.5554147, -6.4989195, -3.5554147, -2.3168416, 2.3168406
3: -10.7896709, -7.8088303, -10.7896709, -7.8088303, -2.7392035, 2.7392039
4: -13.5483303, -10.6108570, -13.5483303, -10.6108570, -2.6516209, 2.6516209
5: -4.6248550, -2.1774464, -4.6248550, -2.1774464, -2.0361338, 2.0361342
6: -4.4857645, -1.9801078, -4.4857645, -1.9801078, -2.2972798, 2.2972798
7: -12.7819271, -8.8199348, -12.7819271, -8.8199348, -3.4429493, 3.4429493
8: -5.4283876, -3.1802011, -5.4283876, -3.1802011, -1.7305593, 1.7305591
9: -1.8567619, 1.0274441, -1.8567619, 1.0274441, -2.8842061, 2.8842061

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103686, upper bound: 1.4111115
time: 6.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111101, upper bound: 1.4111110
time: 5.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7637668, 10.2111721, -2.3236866, 2.3183632
1: -19.2184887, -15.2969704, -19.2533894, -15.2754498, -2.9573779, 2.9700665
2: -6.4989195, -3.5554147, -6.5111165, -3.5523076, -2.3206940, 2.3310363
3: -10.7896709, -7.8088303, -10.8155174, -7.7976027, -2.7506647, 2.7637830
4: -13.5483303, -10.6108570, -13.5691833, -10.5972061, -2.6641822, 2.6723914
5: -4.6248550, -2.1774464, -4.6380377, -2.1673660, -2.0454235, 2.0492222
6: -4.4857645, -1.9801078, -4.5081902, -1.9305443, -2.3470011, 2.3201985
7: -12.7819271, -8.8199348, -12.7976437, -8.7885475, -3.4734030, 3.4564819
8: -5.4283876, -3.1802011, -5.4444995, -3.1584921, -1.7522056, 1.7473409
9: -1.8567619, 1.0274441, -1.9045305, 1.0422702, -2.8990321, 2.9319746

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103686, upper bound: 1.4111106
time: 11.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111101, upper bound: 1.4111114
time: 6.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7782664, 10.1912088, -2.3134170, 2.3304205
1: -19.2248135, -15.2929382, -19.2184887, -15.2969704, -2.9398918, 2.9397917
2: -6.5115609, -3.5520177, -6.4989195, -3.5554147, -2.3384023, 2.3203878
3: -10.7929115, -7.8039832, -10.7896709, -7.8088303, -2.7426081, 2.7438869
4: -13.5695877, -10.6058292, -13.5483303, -10.6108570, -2.6727414, 2.6563778
5: -4.6272163, -2.1694930, -4.6248550, -2.1774464, -2.0384617, 2.0451784
6: -4.4925151, -1.9654703, -4.4857645, -1.9801078, -2.3044987, 2.3080869
7: -12.8078575, -8.8138428, -12.7819271, -8.8199348, -3.4684687, 3.4496899
8: -5.4340730, -3.1679649, -5.4283876, -3.1802011, -1.7415502, 1.7425952
9: -1.8838558, 1.0316219, -1.8567619, 1.0274441, -2.9112999, 2.8883839

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103666, upper bound: 1.4198541
time: 6.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111093, upper bound: 1.4198535
time: 5.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7637668, 10.2111721, -2.3334646, 2.3420556
1: -19.2248135, -15.2929382, -19.2533894, -15.2754498, -2.9615240, 2.9741068
2: -6.5115609, -3.5520177, -6.5111165, -3.5523076, -2.3422661, 2.3345830
3: -10.7929115, -7.8039832, -10.8155174, -7.7976027, -2.7540684, 2.7684841
4: -13.5695877, -10.6058292, -13.5691833, -10.5972061, -2.6853037, 2.6771483
5: -4.6272163, -2.1694930, -4.6380377, -2.1673660, -2.0477524, 2.0582671
6: -4.4925151, -1.9654703, -4.5081902, -1.9305443, -2.3537631, 2.3310051
7: -12.8078575, -8.8138428, -12.7976437, -8.7885475, -3.4988961, 3.4632225
8: -5.4340730, -3.1679649, -5.4444995, -3.1584921, -1.7631965, 1.7593770
9: -1.8838558, 1.0316219, -1.9045305, 1.0422702, -2.9261260, 2.9361525

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103668, upper bound: 1.4198557
time: 6.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111095, upper bound: 1.4198537
time: 5.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7686567, 10.2174139, -2.3304205, 2.3134167
1: -19.2184887, -15.2969704, -19.2248135, -15.2929382, -2.9397917, 2.9398918
2: -6.4989195, -3.5554147, -6.5115609, -3.5520177, -2.3203874, 2.3384023
3: -10.7896709, -7.8088303, -10.7929115, -7.8039832, -2.7438869, 2.7426081
4: -13.5483303, -10.6108570, -13.5695877, -10.6058292, -2.6563778, 2.6727424
5: -4.6248550, -2.1774464, -4.6272163, -2.1694930, -2.0451784, 2.0384619
6: -4.4857645, -1.9801078, -4.4925151, -1.9654703, -2.3080869, 2.3044987
7: -12.7819271, -8.8199348, -12.8078575, -8.8138428, -3.4496894, 3.4684687
8: -5.4283876, -3.1802011, -5.4340730, -3.1679649, -1.7425952, 1.7415500
9: -1.8567619, 1.0274441, -1.8838558, 1.0316219, -2.8883839, 2.9112999

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4191093, upper bound: 1.4111109
time: 5.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198524, upper bound: 1.4111103
time: 5.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7542577, 10.2373705, -2.3435163, 2.3280373
1: -19.2184887, -15.2969704, -19.2597008, -15.2715073, -2.9613371, 2.9741931
2: -6.4989195, -3.5554147, -6.5238338, -3.5489082, -2.3242450, 2.3526840
3: -10.7896709, -7.8088303, -10.8188009, -7.7928100, -2.7553196, 2.7672524
4: -13.5483303, -10.6108570, -13.5904942, -10.5922041, -2.6689730, 2.6935768
5: -4.6248550, -2.1774464, -4.6404004, -2.1595216, -2.0543208, 2.0515530
6: -4.4857645, -1.9801078, -4.5149112, -1.9159023, -2.3509727, 2.3273792
7: -12.7819271, -8.8199348, -12.8235159, -8.7824459, -3.4802017, 3.4820094
8: -5.4283876, -3.1802011, -5.4501781, -3.1462612, -1.7636783, 1.7582679
9: -1.8567619, 1.0274441, -1.9316421, 1.0464482, -2.9032102, 2.9590862

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4191095, upper bound: 1.4111111
time: 5.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198526, upper bound: 1.4111103
time: 5.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7684426, 10.2174129, -2.3221917, 2.3221917
1: -19.2248154, -15.2928429, -19.2248154, -15.2928429, -2.9415922, 2.9415913
2: -6.5115700, -3.5520153, -6.5115700, -3.5520153, -2.3366370, 2.3366373
3: -10.7933302, -7.8039842, -10.7933302, -7.8039842, -2.7630739, 2.7630739
4: -13.5695877, -10.6057549, -13.5695877, -10.6057549, -2.6687698, 2.6687694
5: -4.6272178, -2.1693721, -4.6272178, -2.1693721, -2.0526443, 2.0526443
6: -4.4925199, -1.9654692, -4.4925199, -1.9654692, -2.3094020, 2.3094015
7: -12.8078947, -8.8138428, -12.8078947, -8.8138428, -3.4598813, 3.4598813
8: -5.4340749, -3.1679506, -5.4340749, -3.1679506, -1.7456923, 1.7456923
9: -1.8838563, 1.0316887, -1.8838563, 1.0316887, -2.9155450, 2.9155450

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103668, upper bound: 1.4198547
time: 5.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111094, upper bound: 1.4198541
time: 5.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7540431, 10.2373714, -2.3422370, 2.3368120
1: -19.2248154, -15.2928429, -19.2597008, -15.2714119, -2.9632034, 2.9759111
2: -6.5115700, -3.5520153, -6.5238419, -3.5489097, -2.3405027, 2.3509190
3: -10.7933302, -7.8039842, -10.8192215, -7.7928095, -2.7745051, 2.7876592
4: -13.5695877, -10.6057549, -13.5904942, -10.5921278, -2.6813765, 2.6895542
5: -4.6272178, -2.1693721, -4.6404009, -2.1593976, -2.0618601, 2.0657361
6: -4.4925199, -1.9654692, -4.5149145, -1.9159026, -2.3591394, 2.3323250
7: -12.8078947, -8.8138428, -12.8235540, -8.7824469, -3.4903488, 3.4734478
8: -5.4340749, -3.1679506, -5.4501781, -3.1462479, -1.7673349, 1.7624104
9: -1.8838563, 1.0316887, -1.9316421, 1.0465150, -2.9303713, 2.9633307

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103666, upper bound: 1.4198563
time: 7.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111093, upper bound: 1.4198537
time: 6.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7782664, 10.1912088, -2.3183632, 2.3236871
1: -19.2533894, -15.2754498, -19.2184887, -15.2969704, -2.9700675, 2.9573784
2: -6.5111165, -3.5523076, -6.4989195, -3.5554147, -2.3310361, 2.3206940
3: -10.8155174, -7.7976027, -10.7896709, -7.8088303, -2.7637835, 2.7506642
4: -13.5691833, -10.5972061, -13.5483303, -10.6108570, -2.6723919, 2.6641817
5: -4.6380377, -2.1673660, -4.6248550, -2.1774464, -2.0492225, 2.0454242
6: -4.5081902, -1.9305443, -4.4857645, -1.9801078, -2.3201985, 2.3470011
7: -12.7976437, -8.7885475, -12.7819271, -8.8199348, -3.4564819, 3.4734025
8: -5.4444995, -3.1584921, -5.4283876, -3.1802011, -1.7473412, 1.7522056
9: -1.9045305, 1.0422702, -1.8567619, 1.0274441, -2.9319746, 2.8990321

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103707, upper bound: 1.4183574
time: 6.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111102, upper bound: 1.4183572
time: 5.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7686567, 10.2174139, -2.3420553, 2.3334649
1: -19.2533894, -15.2754498, -19.2248135, -15.2929382, -2.9741068, 2.9615240
2: -6.5111165, -3.5523076, -6.5115609, -3.5520177, -2.3345838, 2.3422656
3: -10.8155174, -7.7976027, -10.7929115, -7.8039832, -2.7684841, 2.7540689
4: -13.5691833, -10.5972061, -13.5695877, -10.6058292, -2.6771488, 2.6853037
5: -4.6380377, -2.1673660, -4.6272163, -2.1694930, -2.0582671, 2.0477519
6: -4.5081902, -1.9305443, -4.4925151, -1.9654703, -2.3310051, 2.3537629
7: -12.7976437, -8.7885475, -12.8078575, -8.8138428, -3.4632230, 3.4988961
8: -5.4444995, -3.1584921, -5.4340730, -3.1679649, -1.7593770, 1.7631965
9: -1.9045305, 1.0422702, -1.8838558, 1.0316219, -2.9361525, 2.9261260

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103686, upper bound: 1.4183575
time: 6.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111101, upper bound: 1.4183569
time: 5.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7542577, 10.2373705, 7.7782664, 10.1912088, -2.3280373, 2.3435163
1: -19.2597008, -15.2715073, -19.2184887, -15.2969704, -2.9741926, 2.9613376
2: -6.5238338, -3.5489082, -6.4989195, -3.5554147, -2.3526835, 2.3242450
3: -10.8188009, -7.7928100, -10.7896709, -7.8088303, -2.7672520, 2.7553196
4: -13.5904942, -10.5922041, -13.5483303, -10.6108570, -2.6935773, 2.6689730
5: -4.6404004, -2.1595216, -4.6248550, -2.1774464, -2.0515528, 2.0543203
6: -4.5149112, -1.9159023, -4.4857645, -1.9801078, -2.3273792, 2.3509729
7: -12.8235159, -8.7824459, -12.7819271, -8.8199348, -3.4820099, 3.4802012
8: -5.4501781, -3.1462612, -5.4283876, -3.1802011, -1.7582681, 1.7636786
9: -1.9316421, 1.0464482, -1.8567619, 1.0274441, -2.9590862, 2.9032102

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4103666, upper bound: 1.4270990
time: 5.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111093, upper bound: 1.4270987
time: 5.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 32.55 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103686, upper bound: 1.4111115
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111101, upper bound: 1.4111110
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103686, upper bound: 1.4111106
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111101, upper bound: 1.4111114
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103666, upper bound: 1.4198541
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111093, upper bound: 1.4198535
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103668, upper bound: 1.4198557
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111095, upper bound: 1.4198537
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4191093, upper bound: 1.4111109
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4198524, upper bound: 1.4111103
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4191095, upper bound: 1.4111111
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4198526, upper bound: 1.4111103
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103668, upper bound: 1.4198547
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111094, upper bound: 1.4198541
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103666, upper bound: 1.4198563
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111093, upper bound: 1.4198537
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103707, upper bound: 1.4183574
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111102, upper bound: 1.4183572
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103686, upper bound: 1.4183575
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111101, upper bound: 1.4183569
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4103666, upper bound: 1.4270990
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 0, lower bound: -1.4111093, upper bound: 1.4270987
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.55
Output dim: 0, lower bound: -1.4111444, upper bound: 1.4271333
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.55
Output dim: 0, lower bound: -1.4137373, upper bound: 1.4209823
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.55
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4209836
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.55
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4297248
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.55
Output dim: 0, lower bound: -1.4137372, upper bound: 1.4297240
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.3750882148742676
rel_dist={0: [-1.4297502739106367, 1.4297521247570089]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565350, upper bound: 1.0523701
time: 7.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573862, upper bound: 1.0573857
time: 8.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.63
Output dim: 0, lower bound: -1.0565350, upper bound: 1.0523701
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.63
Output dim: 0, lower bound: -1.0573862, upper bound: 1.0573857

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 7.7684388, 10.2174206, 7.7576704, 10.2280102, -2.0955739, 2.0953202
1: -19.2248173, -15.2928391, -19.2448311, -15.2768841, -2.5382566, 2.5416245
2: -6.5115714, -3.5520163, -6.5188708, -3.5503392, -2.0200911, 2.0262465
3: -10.7933292, -7.8039856, -10.8083515, -7.7941904, -2.4436517, 2.4418607
4: -13.5695906, -10.6057529, -13.5810127, -10.5955544, -2.3098898, 2.3116002
5: -4.6272192, -2.1693683, -4.6342835, -2.1624115, -1.7901859, 1.7891240
6: -4.4925184, -1.9654677, -4.5130796, -1.9392797, -2.1051545, 2.0995598
7: -12.8079023, -8.8138399, -12.8211451, -8.7960434, -3.0725403, 3.0719500
8: -5.4340787, -3.1679492, -5.4470539, -3.1562667, -1.5038319, 1.5053551
9: -1.8838596, 1.0316887, -1.9102111, 1.0459359, -2.7210627, 2.7329164

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512839, upper bound: 1.0520865
time: 5.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565238, upper bound: 1.0523572
time: 8.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7540388, 10.2373838, -2.1192141, 2.1073723
1: -19.2597046, -15.2714100, -19.2597103, -15.2714062, -2.5652900, 2.5779567
2: -6.5238457, -3.5489109, -6.5238481, -3.5489058, -2.0367079, 2.0370188
3: -10.8192225, -7.7928071, -10.8192282, -7.7928076, -2.4687619, 2.4773293
4: -13.5904970, -10.5921268, -13.5905037, -10.5921240, -2.3196297, 2.3334045
5: -4.6404028, -2.1593966, -4.6404047, -2.1593952, -1.8031054, 1.8048120
6: -4.5149150, -1.9159002, -4.5149159, -1.9158906, -2.1515875, 2.1066718
7: -12.8235588, -8.7824440, -12.8235607, -8.7824383, -3.1019778, 3.0845456
8: -5.4501791, -3.1462469, -5.4501820, -3.1462431, -1.5305820, 1.5110490
9: -1.9316473, 1.0465150, -1.9316545, 1.0465150, -2.7490501, 2.7701054

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521421, upper bound: 1.0571213
time: 8.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573749, upper bound: 1.0573750
time: 6.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 29.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.13
Output dim: 0, lower bound: -1.0512839, upper bound: 1.0520865
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.13
Output dim: 0, lower bound: -1.0565238, upper bound: 1.0523572
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.13
Output dim: 0, lower bound: -1.0521421, upper bound: 1.0571213
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.13
Output dim: 0, lower bound: -1.0573749, upper bound: 1.0573750

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 7.7702370, 10.2073498, 7.7674398, 10.2018032, -2.0669155, 2.0752439
1: -19.2222309, -15.2938881, -19.2385197, -15.2809639, -2.5308323, 2.5383048
2: -6.5065241, -3.5525074, -6.5061755, -3.5537386, -2.0114951, 2.0041101
3: -10.7924032, -7.8054309, -10.8046350, -7.7990079, -2.4357147, 2.4344897
4: -13.5617371, -10.6064196, -13.5597210, -10.6006479, -2.2950315, 2.2898412
5: -4.6266246, -2.1723967, -4.6319170, -2.1704311, -1.7799978, 1.7829766
6: -4.4906611, -1.9710909, -4.5063367, -1.9539196, -2.0915227, 2.0851364
7: -12.7977867, -8.8146200, -12.7952328, -8.8021507, -3.0550070, 3.0452976
8: -5.4326797, -3.1724625, -5.4413605, -3.1685114, -1.4901617, 1.4902389
9: -1.8735580, 1.0315781, -1.8831115, 1.0416906, -2.7072506, 2.7054968

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512838, upper bound: 1.0471111
time: 6.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512838, upper bound: 1.0520881
time: 5.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 7.7684388, 10.2174206, 7.7576723, 10.2280045, -2.0741487, 2.0953183
1: -19.2248173, -15.2928391, -19.2448311, -15.2768831, -2.5381126, 2.5380197
2: -6.5115714, -3.5520163, -6.5188684, -3.5503397, -2.0200906, 2.0199754
3: -10.7933292, -7.8039856, -10.8083506, -7.7941914, -2.4535856, 2.4400702
4: -13.5695906, -10.6057529, -13.5810108, -10.5955544, -2.3093238, 2.2991457
5: -4.6272192, -2.1693683, -4.6342840, -2.1624134, -1.7895885, 1.7923810
6: -4.4925184, -1.9654677, -4.5130773, -1.9392816, -2.0963550, 2.0988884
7: -12.8079023, -8.8138399, -12.8211403, -8.7960443, -3.0723076, 3.0532055
8: -5.4340787, -3.1679492, -5.4470520, -3.1562681, -1.4945168, 1.5053542
9: -1.8838596, 1.0316887, -1.9102092, 1.0459368, -2.7210617, 2.7289047

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0562603, upper bound: 1.0471093
time: 6.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0562603, upper bound: 1.0523595
time: 7.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 7.7558002, 10.2273102, 7.7637639, 10.2111769, -2.0905962, 2.0872216
1: -19.2571220, -15.2724323, -19.2533970, -15.2754459, -2.5578794, 2.5746536
2: -6.5187736, -3.5493991, -6.5111213, -3.5523076, -2.0281048, 2.0148358
3: -10.8182745, -7.7942362, -10.8155212, -7.7976031, -2.4608006, 2.4699764
4: -13.5826235, -10.5927801, -13.5691862, -10.5972042, -2.3047299, 2.3116102
5: -4.6398072, -2.1623821, -4.6380420, -2.1673646, -1.7928748, 1.7987070
6: -4.5130687, -1.9215262, -4.5081921, -1.9305341, -2.1379552, 2.0922527
7: -12.8134775, -8.7832298, -12.7976437, -8.7885408, -3.0844579, 3.0579605
8: -5.4488106, -3.1507573, -5.4445038, -3.1584902, -1.5169308, 1.4959404
9: -1.9213347, 1.0464048, -1.9045386, 1.0422699, -2.7352095, 2.7426534

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516120, upper bound: 1.0514485
time: 5.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521384, upper bound: 1.0571152
time: 13.09 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7540417, 10.2373743, -2.0977898, 2.1073723
1: -19.2597046, -15.2714100, -19.2597103, -15.2714081, -2.5651407, 2.5744014
2: -6.5238457, -3.5489109, -6.5238471, -3.5489075, -2.0367074, 2.0307465
3: -10.8192225, -7.7928071, -10.8192272, -7.7928071, -2.4786272, 2.4755383
4: -13.5904970, -10.5921268, -13.5905018, -10.5921230, -2.3190613, 2.3209348
5: -4.6404028, -2.1593966, -4.6404047, -2.1593971, -1.8025007, 1.8081317
6: -4.5149150, -1.9159002, -4.5149155, -1.9158924, -2.1428032, 2.1059947
7: -12.8235588, -8.7824440, -12.8235569, -8.7824402, -3.1017427, 3.0658193
8: -5.4501791, -3.1462469, -5.4501805, -3.1462479, -1.5212674, 1.5110478
9: -1.9316473, 1.0465150, -1.9316511, 1.0465145, -2.7490511, 2.7660928

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0571198, upper bound: 1.0521421
time: 5.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0571198, upper bound: 1.0573753
time: 6.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0512838, upper bound: 1.0471111
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0512838, upper bound: 1.0520881
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0562603, upper bound: 1.0471093
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0562603, upper bound: 1.0523595
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0516120, upper bound: 1.0514485
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0521384, upper bound: 1.0571152
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0571198, upper bound: 1.0521421
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.20
Output dim: 0, lower bound: -1.0571198, upper bound: 1.0573753

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7674398, 10.2018032, -2.0587273, 2.0585370
1: -19.2184887, -15.2969704, -19.2385197, -15.2809639, -2.5318809, 2.5352273
2: -6.4989195, -3.5554147, -6.5061755, -3.5537386, -1.9949951, 2.0011094
3: -10.7896709, -7.8088303, -10.8046350, -7.7990079, -2.4317093, 2.4298348
4: -13.5483303, -10.6108570, -13.5597210, -10.6006479, -2.2833328, 2.2850351
5: -4.6248550, -2.1774464, -4.6319170, -2.1704311, -1.7774849, 1.7763405
6: -4.4857645, -1.9801078, -4.5063367, -1.9539196, -2.0857148, 2.0801282
7: -12.7819271, -8.8199348, -12.7952328, -8.8021507, -3.0402746, 3.0397120
8: -5.4283876, -3.1802011, -5.4413605, -3.1685114, -1.4812155, 1.4827819
9: -1.8567619, 1.0274441, -1.8831115, 1.0416906, -2.6904936, 2.7023010

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471111
time: 7.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471106
time: 13.89 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7674398, 10.2018032, -2.0685053, 2.0807097
1: -19.2248135, -15.2929382, -19.2385197, -15.2809639, -2.5339050, 2.5392675
2: -6.5115609, -3.5520177, -6.5061755, -3.5537386, -2.0165219, 2.0046570
3: -10.7929115, -7.8039832, -10.8046350, -7.7990079, -2.4351139, 2.4340887
4: -13.5695877, -10.6058292, -13.5597210, -10.6006479, -2.3044553, 2.2897921
5: -4.6272163, -2.1694930, -4.6319170, -2.1704311, -1.7798128, 1.7854204
6: -4.4925151, -1.9654703, -4.5063367, -1.9539196, -2.0929332, 2.0915699
7: -12.8078575, -8.8138428, -12.7952328, -8.8021507, -3.0654678, 3.0464525
8: -5.4340730, -3.1679649, -5.4413605, -3.1685114, -1.4917777, 1.4948180
9: -1.8838558, 1.0316219, -1.8831115, 1.0416906, -2.7176647, 2.7056227

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0520862
time: 5.71 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0520881
time: 7.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7578874, 10.2280006, -2.0766978, 2.0682502
1: -19.2184887, -15.2969704, -19.2448311, -15.2769823, -2.5358772, 2.5372314
2: -6.4989195, -3.5554147, -6.5188599, -3.5503378, -1.9985476, 2.0226812
3: -10.7896709, -7.8088303, -10.8079357, -7.7941914, -2.4359426, 2.4333057
4: -13.5483303, -10.6108570, -13.5810070, -10.5956316, -2.2881193, 2.3061953
5: -4.6248550, -2.1774464, -4.6342835, -2.1625376, -1.7864895, 1.7786794
6: -4.4857645, -1.9801078, -4.5130763, -1.9392821, -2.0971565, 2.0873294
7: -12.7819271, -8.8199348, -12.8211040, -8.7960463, -3.0470419, 3.0649085
8: -5.4283876, -3.1802011, -5.4470520, -3.1562819, -1.4932463, 1.4932959
9: -1.8567619, 1.0274441, -1.9102063, 1.0458698, -2.6938152, 2.7295203

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471094
time: 5.54 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471112
time: 6.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7576723, 10.2280045, -2.0741472, 2.0738945
1: -19.2248154, -15.2928429, -19.2448311, -15.2768831, -2.5346556, 2.5380173
2: -6.5115700, -3.5520153, -6.5188684, -3.5503397, -2.0138187, 2.0199754
3: -10.7933302, -7.8039842, -10.8083506, -7.7941914, -2.4535828, 2.4517107
4: -13.5695877, -10.6057549, -13.5810108, -10.5955544, -2.2974591, 2.2991433
5: -4.6272178, -2.1693721, -4.6342840, -2.1624134, -1.7934628, 1.7923787
6: -4.4925199, -1.9654692, -4.5130773, -1.9392816, -2.0963516, 2.0907393
7: -12.8078947, -8.8138428, -12.8211403, -8.7960443, -3.0537663, 3.0532041
8: -5.4340749, -3.1679506, -5.4470520, -3.1562681, -1.4945154, 1.4960394
9: -1.8838563, 1.0316887, -1.9102092, 1.0459368, -2.7170448, 2.7289038

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471091
time: 36.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477551, upper bound: 1.0471112
time: 7.99 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 7.7574477, 10.2216148, 7.7645802, 10.2083664, -2.0861449, 2.0803022
1: -19.2507896, -15.2749195, -19.2502766, -15.2766771, -2.5501661, 2.5686860
2: -6.5174265, -3.5529709, -6.5104589, -3.5551786, -2.0235515, 2.0119276
3: -10.8088884, -7.7955608, -10.8108921, -7.7982597, -2.4503074, 2.4636149
4: -13.5770559, -10.5936604, -13.5664454, -10.5976410, -2.2981234, 2.3076496
5: -4.6351199, -2.1639671, -4.6357288, -2.1681526, -1.7875876, 1.7942977
6: -4.5109863, -1.9384058, -4.5071635, -1.9388540, -2.1269603, 2.0740423
7: -12.8125744, -8.7891464, -12.7971992, -8.7914581, -3.0799818, 3.0549679
8: -5.4473448, -3.1628408, -5.4437780, -3.1644435, -1.5092831, 1.4829938
9: -1.9086185, 1.0460641, -1.8982725, 1.0421031, -2.7221870, 2.7359548

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0464712, upper bound: 1.0514480
time: 7.75 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0464711, upper bound: 1.0514501
time: 6.30 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 7.7503729, 10.2277851, 7.7637749, 10.2111702, -2.0964375, 2.0847430
1: -19.2595863, -15.2657261, -19.2533741, -15.2754469, -2.5571866, 2.5812049
2: -6.5211234, -3.5484221, -6.5111179, -3.5524006, -2.0301166, 2.0154979
3: -10.8212938, -7.7845922, -10.8155136, -7.7977018, -2.4616799, 2.4798164
4: -13.5830469, -10.5877218, -13.5689669, -10.5972061, -2.3022952, 2.3163118
5: -4.6401806, -2.1582875, -4.6379538, -2.1673663, -1.7930541, 1.8027799
6: -4.5311103, -1.9204981, -4.5081840, -1.9305494, -2.1453705, 2.0838976
7: -12.8199348, -8.7819977, -12.7976265, -8.7885494, -3.0906734, 3.0624075
8: -5.4624352, -3.1498737, -5.4445000, -3.1585793, -1.5247068, 1.4902480
9: -1.9240503, 1.0569043, -1.9045300, 1.0422432, -2.7352877, 2.7533731

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0464711, upper bound: 1.0565897
time: 10.33 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0464712, upper bound: 1.0565915
time: 7.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7542567, 10.2373753, -2.1000686, 2.0803056
1: -19.2533894, -15.2754498, -19.2597065, -15.2715034, -2.5629196, 2.5736332
2: -6.5111165, -3.5523076, -6.5238357, -3.5489082, -2.0150719, 2.0334511
3: -10.8155174, -7.7976027, -10.8188076, -7.7928076, -2.4609976, 2.4688029
4: -13.5691833, -10.5972061, -13.5905018, -10.5922041, -2.2977929, 2.3279638
5: -4.6380377, -2.1673660, -4.6404033, -2.1595194, -1.7993965, 1.7945032
6: -4.5081902, -1.9305443, -4.5149131, -1.9158934, -2.1394548, 2.0944276
7: -12.7976437, -8.7885475, -12.8235178, -8.7824402, -3.0765419, 3.0775332
8: -5.4444995, -3.1584921, -5.4501805, -3.1462612, -1.5188243, 1.4989924
9: -1.9045305, 1.0422702, -1.9316487, 1.0464494, -2.7217245, 2.7667084

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0512833
time: 5.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0521439
time: 5.67 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7540431, 10.2373714, 7.7540417, 10.2373743, -2.0977893, 2.0859475
1: -19.2597008, -15.2714119, -19.2597103, -15.2714081, -2.5617337, 2.5744004
2: -6.5238419, -3.5489097, -6.5238471, -3.5489075, -2.0304356, 2.0307465
3: -10.8192215, -7.7928095, -10.8192272, -7.7928071, -2.4786243, 2.4871917
4: -13.5904942, -10.5921278, -13.5905018, -10.5921230, -2.3071594, 2.3209338
5: -4.6404009, -2.1593976, -4.6404047, -2.1593971, -1.8064208, 1.8081288
6: -4.5149145, -1.9159026, -4.5149155, -1.9158924, -2.1427999, 2.0978823
7: -12.8235540, -8.7824469, -12.8235569, -8.7824402, -3.0832500, 3.0658178
8: -5.4501781, -3.1462479, -5.4501805, -3.1462479, -1.5212660, 1.5017333
9: -1.9316421, 1.0465150, -1.9316511, 1.0465145, -2.7450390, 2.7660928

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0565239
time: 11.87 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0565262
time: 7.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471111
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471106
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0520862
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0520881
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471094
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471112
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477552, upper bound: 1.0471091
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0477551, upper bound: 1.0471112
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0464712, upper bound: 1.0514480
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0464711, upper bound: 1.0514501
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0464711, upper bound: 1.0565897
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0464712, upper bound: 1.0565915
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0512833
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0521439
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0565239
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.24
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0565262

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7782664, 10.1912088, -2.0476961, 2.0476961
1: -19.2184887, -15.2969704, -19.2184887, -15.2969704, -2.5156760, 2.5156755
2: -6.4989195, -3.5554147, -6.4989195, -3.5554147, -1.9934716, 1.9934721
3: -10.7896709, -7.8088303, -10.7896709, -7.8088303, -2.4190216, 2.4190211
4: -13.5483303, -10.6108570, -13.5483303, -10.6108570, -2.2734394, 2.2734394
5: -4.6248550, -2.1774464, -4.6248550, -2.1774464, -1.7697926, 1.7697923
6: -4.4857645, -1.9801078, -4.4857645, -1.9801078, -2.0595131, 2.0595131
7: -12.7819271, -8.8199348, -12.7819271, -8.8199348, -3.0257483, 3.0257487
8: -5.4283876, -3.1802011, -5.4283876, -3.1802011, -1.4695964, 1.4695961
9: -1.8567619, 1.0274441, -1.8567619, 1.0274441, -2.6758070, 2.6758070

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0469808, upper bound: 1.0470848
time: 7.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477300, upper bound: 1.0470847
time: 5.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7637668, 10.2111721, -2.0677443, 2.0624204
1: -19.2184887, -15.2969704, -19.2533894, -15.2754498, -2.5373025, 2.5499907
2: -6.4989195, -3.5554147, -6.5111165, -3.5523076, -1.9973249, 2.0076673
3: -10.7896709, -7.8088303, -10.8155174, -7.7976027, -2.4304819, 2.4436002
4: -13.5483303, -10.6108570, -13.5691833, -10.5972061, -2.2860007, 2.2942104
5: -4.6248550, -2.1774464, -4.6380377, -2.1673660, -1.7790823, 1.7828803
6: -4.4857645, -1.9801078, -4.5081902, -1.9305443, -2.1001439, 2.0824313
7: -12.7819271, -8.8199348, -12.7976437, -8.7885475, -3.0562019, 3.0392818
8: -5.4283876, -3.1802011, -5.4444995, -3.1584921, -1.4912426, 1.4863780
9: -1.8567619, 1.0274441, -1.9045305, 1.0422702, -2.6912260, 2.7240291

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0469808, upper bound: 1.0470847
time: 9.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477300, upper bound: 1.0470849
time: 6.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7782664, 10.1912088, -2.0574737, 2.0691957
1: -19.2248135, -15.2929382, -19.2184887, -15.2969704, -2.5176926, 2.5197153
2: -6.5115609, -3.5520177, -6.4989195, -3.5554147, -2.0149932, 1.9970193
3: -10.7929115, -7.8039832, -10.7896709, -7.8088303, -2.4224253, 2.4232659
4: -13.5695877, -10.6058292, -13.5483303, -10.6108570, -2.2945623, 2.2781963
5: -4.6272163, -2.1694930, -4.6248550, -2.1774464, -1.7721205, 1.7788715
6: -4.4925151, -1.9654703, -4.4857645, -1.9801078, -2.0667315, 2.0709548
7: -12.8078575, -8.8138428, -12.7819271, -8.8199348, -3.0509434, 3.0324888
8: -5.4340730, -3.1679649, -5.4283876, -3.1802011, -1.4801586, 1.4816322
9: -1.8838558, 1.0316219, -1.8567619, 1.0274441, -2.7029772, 2.6791286

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0469809, upper bound: 1.0520607
time: 7.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477292, upper bound: 1.0520607
time: 5.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7637668, 10.2111721, -2.0772133, 2.0800176
1: -19.2248135, -15.2929382, -19.2533894, -15.2754498, -2.5393248, 2.5540309
2: -6.5115609, -3.5520177, -6.5111165, -3.5523076, -2.0188560, 2.0112145
3: -10.7929115, -7.8039832, -10.8155174, -7.7976027, -2.4338856, 2.4478626
4: -13.5695877, -10.6058292, -13.5691833, -10.5972061, -2.3071237, 2.2989674
5: -4.6272163, -2.1694930, -4.6380377, -2.1673660, -1.7814102, 1.7919600
6: -4.4925151, -1.9654703, -4.5081902, -1.9305443, -2.1052737, 2.0896845
7: -12.8078575, -8.8138428, -12.7976437, -8.7885475, -3.0813708, 3.0460219
8: -5.4340730, -3.1679649, -5.4444995, -3.1584921, -1.5003428, 1.4971805
9: -1.8838558, 1.0316219, -1.9045305, 1.0422702, -2.7183962, 2.7273507

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0469789, upper bound: 1.0520628
time: 8.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477292, upper bound: 1.0520625
time: 9.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7686567, 10.2174139, -2.0691957, 2.0574737
1: -19.2184887, -15.2969704, -19.2248135, -15.2929382, -2.5197153, 2.5176930
2: -6.4989195, -3.5554147, -6.5115609, -3.5520177, -1.9970193, 2.0149932
3: -10.7896709, -7.8088303, -10.7929115, -7.8039832, -2.4232655, 2.4224253
4: -13.5483303, -10.6108570, -13.5695877, -10.6058292, -2.2781963, 2.2945623
5: -4.6248550, -2.1774464, -4.6272163, -2.1694930, -1.7788715, 1.7721200
6: -4.4857645, -1.9801078, -4.4925151, -1.9654703, -2.0709548, 2.0667315
7: -12.7819271, -8.8199348, -12.8078575, -8.8138428, -3.0324893, 3.0509434
8: -5.4283876, -3.1802011, -5.4340730, -3.1679649, -1.4816322, 1.4801586
9: -1.8567619, 1.0274441, -1.8838558, 1.0316219, -2.6791286, 2.7029777

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519551, upper bound: 1.0470843
time: 5.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527059, upper bound: 1.0470841
time: 5.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7542577, 10.2373705, -2.0790768, 2.0720944
1: -19.2184887, -15.2969704, -19.2597008, -15.2715073, -2.5412617, 2.5519943
2: -6.4989195, -3.5554147, -6.5238338, -3.5489082, -2.0008759, 2.0292749
3: -10.7896709, -7.8088303, -10.8188009, -7.7928100, -2.4346981, 2.4470696
4: -13.5483303, -10.6108570, -13.5904942, -10.5922041, -2.2907920, 2.3153973
5: -4.6248550, -2.1774464, -4.6404004, -2.1595216, -1.7880139, 1.7852111
6: -4.4857645, -1.9801078, -4.5149112, -1.9159023, -2.1011953, 2.0896120
7: -12.7819271, -8.8199348, -12.8235159, -8.7824459, -3.0630007, 3.0644841
8: -5.4283876, -3.1802011, -5.4501781, -3.1462612, -1.4956310, 1.4968765
9: -1.8567619, 1.0274441, -1.9316421, 1.0464482, -2.6945467, 2.7469335

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519551, upper bound: 1.0470840
time: 14.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527059, upper bound: 1.0470834
time: 7.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7684426, 10.2174129, -2.0631185, 2.0631185
1: -19.2248154, -15.2928429, -19.2248154, -15.2928429, -2.5184603, 2.5184593
2: -6.5115700, -3.5520153, -6.5115700, -3.5520153, -2.0122867, 2.0122869
3: -10.7933302, -7.8039842, -10.7933302, -7.8039842, -2.4409056, 2.4409051
4: -13.5695877, -10.6057549, -13.5695877, -10.6057549, -2.2875366, 2.2875366
5: -4.6272178, -2.1693721, -4.6272178, -2.1693721, -1.7858186, 1.7858188
6: -4.4925199, -1.9654692, -4.4925199, -1.9654692, -2.0701280, 2.0701280
7: -12.8078947, -8.8138428, -12.8078947, -8.8138428, -3.0392060, 3.0392056
8: -5.4340749, -3.1679506, -5.4340749, -3.1679506, -1.4829016, 1.4829021
9: -1.8838563, 1.0316887, -1.8838563, 1.0316887, -2.7023592, 2.7023592

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0473116, upper bound: 1.0523400
time: 8.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0480694, upper bound: 1.0523397
time: 6.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 36.34 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0469808, upper bound: 1.0470848
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0477300, upper bound: 1.0470847
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0469808, upper bound: 1.0470847
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0477300, upper bound: 1.0470849
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0469809, upper bound: 1.0520607
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0477292, upper bound: 1.0520607
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0469789, upper bound: 1.0520628
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0477292, upper bound: 1.0520625
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0519551, upper bound: 1.0470843
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0527059, upper bound: 1.0470841
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0519551, upper bound: 1.0470840
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0527059, upper bound: 1.0470834
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0473116, upper bound: 1.0523400
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.34
Output dim: 0, lower bound: -1.0480694, upper bound: 1.0523397
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0477551, upper bound: 1.0471112
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0464712, upper bound: 1.0514480
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0464711, upper bound: 1.0514501
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0464711, upper bound: 1.0565897
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0464712, upper bound: 1.0565915
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0512833
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0521439
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0565239
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.34
Output dim: 0, lower bound: -1.0471093, upper bound: 1.0565262
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.1192193031311035
rel_dist={0: [-1.0573893980579818, 1.057389512159725]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092265, upper bound: 0.9062294
time: 6.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094649, upper bound: 0.9094642
time: 5.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.41
Output dim: 0, lower bound: -0.9092265, upper bound: 0.9062294
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.41
Output dim: 0, lower bound: -0.9094649, upper bound: 0.9094642

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 7.7684388, 10.2174206, 7.7583423, 10.2263012, -2.0086422, 2.0093212
1: -19.2248173, -15.2928391, -19.2421169, -15.2778988, -2.3965268, 2.3981981
2: -6.5115714, -3.5520163, -6.5179663, -3.5506077, -1.9118509, 1.9175258
3: -10.7933292, -7.8039856, -10.8063602, -7.7944450, -2.3365154, 2.3324633
4: -13.5695906, -10.6057529, -13.5792751, -10.5961924, -2.1831961, 2.1838579
5: -4.6272192, -2.1693683, -4.6331682, -2.1629679, -1.7008157, 1.6991425
6: -4.4925184, -1.9654677, -4.5127406, -1.9435682, -2.0217953, 2.0200939
7: -12.8079023, -8.8138399, -12.8206997, -8.7985287, -2.9304328, 2.9324141
8: -5.4340787, -3.1679492, -5.4464722, -3.1581059, -1.4148607, 1.4175708
9: -1.8838596, 1.0316887, -1.9062834, 1.0458262, -2.6496601, 2.6576662

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9052308, upper bound: 0.9059428
time: 8.10 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092179, upper bound: 0.9062214
time: 5.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7540402, 10.2373829, -2.0339231, 2.0214620
1: -19.2597046, -15.2714100, -19.2597084, -15.2714071, -2.4239235, 2.4372220
2: -6.5238457, -3.5489109, -6.5238466, -3.5489068, -1.9289041, 1.9291377
3: -10.8192225, -7.7928071, -10.8192253, -7.7928085, -2.3615279, 2.3700886
4: -13.5904970, -10.5921268, -13.5905037, -10.5921230, -2.1927261, 2.2073431
5: -4.6404028, -2.1593966, -4.6404037, -2.1593943, -1.7138948, 1.7160425
6: -4.5149150, -1.9159002, -4.5149169, -1.9158931, -2.0725417, 2.0253801
7: -12.8235588, -8.7824440, -12.8235588, -8.7824402, -2.9628010, 2.9439578
8: -5.4501791, -3.1462469, -5.4501810, -3.1462440, -1.4434509, 1.4229410
9: -1.9316473, 1.0465150, -1.9316535, 1.0465147, -2.6767368, 2.6988397

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054692, upper bound: 0.9091776
time: 5.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094563, upper bound: 0.9094584
time: 6.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.78 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 26.78
Output dim: 0, lower bound: -0.9052308, upper bound: 0.9059428
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 0, lower bound: -0.9092179, upper bound: 0.9062214
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 0, lower bound: -0.9054692, upper bound: 0.9091776
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.78
Output dim: 0, lower bound: -0.9094563, upper bound: 0.9094584

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 7.7684436, 10.2174206, 7.7583432, 10.2262955, -1.9861498, 2.0093198
1: -19.2248135, -15.2928410, -19.2421112, -15.2778988, -2.3963842, 2.3942814
2: -6.5115714, -3.5520165, -6.5179648, -3.5506089, -1.9118509, 1.9109390
3: -10.7933292, -7.8039846, -10.8063583, -7.7944460, -2.3459315, 2.3303504
4: -13.5695896, -10.6057539, -13.5792732, -10.5961924, -2.1826291, 2.1703901
5: -4.6272182, -2.1693690, -4.6331673, -2.1629708, -1.7001114, 1.7022257
6: -4.4925199, -1.9654685, -4.5127382, -1.9435723, -2.0122824, 2.0194225
7: -12.8079023, -8.8138409, -12.8206949, -8.7985306, -2.9301996, 2.9126205
8: -5.4340734, -3.1679506, -5.4464731, -3.1581078, -1.4050796, 1.4175694
9: -1.8838596, 1.0316887, -1.9062791, 1.0458276, -2.6496582, 2.6534519

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9089403, upper bound: 0.9022335
time: 10.92 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9089403, upper bound: 0.9062213
time: 8.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 7.7561893, 10.2251034, 7.7637653, 10.2111740, -2.0048909, 1.9990869
1: -19.2565536, -15.2726507, -19.2533970, -15.2754459, -2.4158401, 2.4343972
2: -6.5176897, -3.5495093, -6.5111198, -3.5523093, -1.9191980, 1.9068465
3: -10.8180628, -7.7945485, -10.8155184, -7.7976017, -2.3534226, 2.3624020
4: -13.5808964, -10.5929270, -13.5691872, -10.5972052, -2.1757512, 2.1854157
5: -4.6396770, -2.1630464, -4.6380405, -2.1673656, -1.7035298, 1.7092443
6: -4.5126581, -1.9227517, -4.5081921, -1.9305363, -2.0582576, 2.0095482
7: -12.8113174, -8.7834044, -12.7976437, -8.7885437, -2.9429631, 2.9172111
8: -5.4485092, -3.1517334, -5.4445038, -3.1584902, -1.4294465, 1.4069622
9: -1.9190750, 1.0463793, -1.9045362, 1.0422704, -2.6605864, 2.6713438

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9052835, upper bound: 0.9053066
time: 7.40 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054666, upper bound: 0.9091747
time: 5.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 7.7540421, 10.2373762, 7.7540421, 10.2373772, -2.0114303, 2.0214605
1: -19.2597065, -15.2714100, -19.2597084, -15.2714081, -2.4237747, 2.4333539
2: -6.5238438, -3.5489097, -6.5238442, -3.5489092, -1.9289026, 1.9225519
3: -10.8192215, -7.7928076, -10.8192244, -7.7928085, -2.3708773, 2.3679729
4: -13.5904980, -10.5921278, -13.5904999, -10.5921240, -2.1921563, 2.1938572
5: -4.6404009, -2.1593971, -4.6404033, -2.1593971, -1.7131805, 1.7191885
6: -4.5149145, -1.9159007, -4.5149150, -1.9158943, -2.0630412, 2.0247035
7: -12.8235588, -8.7824450, -12.8235579, -8.7824392, -2.9625645, 2.9241800
8: -5.4501791, -3.1462483, -5.4501796, -3.1462474, -1.4336698, 1.4229386
9: -1.9316459, 1.0465150, -1.9316492, 1.0465145, -2.6767349, 2.6946259

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091786, upper bound: 0.9054681
time: 5.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091786, upper bound: 0.9054682
time: 6.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.66 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.66
Output dim: 0, lower bound: -0.9089403, upper bound: 0.9022335
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.66
Output dim: 0, lower bound: -0.9089403, upper bound: 0.9062213
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 26.66
Output dim: 0, lower bound: -0.9052835, upper bound: 0.9053066
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.66
Output dim: 0, lower bound: -0.9054666, upper bound: 0.9091747
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.66
Output dim: 0, lower bound: -0.9091786, upper bound: 0.9054681
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.66
Output dim: 0, lower bound: -0.9091786, upper bound: 0.9054682

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7585583, 10.2262955, -1.9881196, 1.9822259
1: -19.2184887, -15.2969704, -19.2421131, -15.2779951, -2.3948555, 2.3938046
2: -6.4989195, -3.5554147, -6.5179577, -3.5506091, -1.8903222, 1.9139588
3: -10.7896709, -7.8088303, -10.8059444, -7.7944455, -2.3288064, 2.3240576
4: -13.5483303, -10.6108570, -13.5792732, -10.5962687, -2.1614251, 2.1784530
5: -4.6248550, -2.1774464, -4.6331663, -2.1630921, -1.6971207, 1.6886857
6: -4.4857645, -1.9801078, -4.5127363, -1.9435716, -2.0137968, 2.0076547
7: -12.7819271, -8.8199348, -12.8206558, -8.7985306, -2.9050431, 2.9253731
8: -5.4283876, -3.1802011, -5.4464731, -3.1581211, -1.4044182, 1.4055111
9: -1.8567619, 1.0274441, -1.9062772, 1.0457590, -2.6224108, 2.6542478

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028991, upper bound: 0.9022334
time: 9.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028991, upper bound: 0.9022334
time: 5.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7583432, 10.2262955, -1.9861484, 1.9868269
1: -19.2248154, -15.2928429, -19.2421112, -15.2778988, -2.3926096, 2.3942800
2: -6.5115700, -3.5520153, -6.5179648, -3.5506089, -1.9052653, 1.9109390
3: -10.7933302, -7.8039842, -10.8063583, -7.7944460, -2.3459287, 2.3417969
4: -13.5695877, -10.6057549, -13.5792732, -10.5961924, -2.1697450, 2.1703882
5: -4.6272178, -2.1693721, -4.6331673, -2.1629708, -1.7039118, 1.7022243
6: -4.4925199, -1.9654692, -4.5127382, -1.9435723, -2.0122795, 2.0105553
7: -12.8078947, -8.8138428, -12.8206949, -8.7985306, -2.9106040, 2.9126201
8: -5.4340749, -3.1679506, -5.4464731, -3.1581078, -1.4050782, 1.4077883
9: -1.8838563, 1.0316887, -1.9062791, 1.0458276, -2.6454430, 2.6534510

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028990, upper bound: 0.9022337
time: 5.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9028991, upper bound: 0.9022335
time: 7.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 7.7507610, 10.2255783, 7.7637749, 10.2111664, -2.0107327, 1.9961858
1: -19.2590218, -15.2659531, -19.2533703, -15.2754498, -2.4147377, 2.4409428
2: -6.5200481, -3.5485327, -6.5111156, -3.5524170, -1.9211984, 1.9071748
3: -10.8210812, -7.7849112, -10.8155107, -7.7977214, -2.3541675, 2.3722396
4: -13.5813208, -10.5878696, -13.5689240, -10.5972061, -2.1729074, 2.1900735
5: -4.6400499, -2.1589501, -4.6379385, -2.1673663, -1.7033625, 1.7132974
6: -4.5306988, -1.9217235, -4.5081825, -1.9305550, -2.0646257, 1.9999309
7: -12.8177814, -8.7821751, -12.7976265, -8.7885513, -2.9491749, 2.9211826
8: -5.4621334, -3.1508503, -5.4445019, -3.1585970, -1.4364264, 1.4004004
9: -1.9217916, 1.0568781, -1.9045267, 1.0422378, -2.6603193, 2.6820612

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9089922
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9091752
time: 6.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7542562, 10.2373734, -2.0127234, 1.9943695
1: -19.2533894, -15.2754498, -19.2597065, -15.2715044, -2.4222612, 2.4329004
2: -6.5111165, -3.5523076, -6.5238361, -3.5489101, -1.9072824, 1.9255695
3: -10.8155174, -7.7976027, -10.8188066, -7.7928081, -2.3537626, 2.3617077
4: -13.5691833, -10.5972061, -13.5904999, -10.5922031, -2.1708889, 2.2019024
5: -4.6380377, -2.1673660, -4.6404028, -2.1595211, -1.7101870, 1.7057221
6: -4.5081902, -1.9305443, -4.5149131, -1.9158949, -2.0587096, 2.0129247
7: -12.7976437, -8.7885475, -12.8235178, -8.7824402, -2.9374733, 2.9369464
8: -5.4444995, -3.1584921, -5.4501791, -3.1462617, -1.4305573, 1.4108844
9: -1.9045305, 1.0422702, -1.9316487, 1.0464478, -2.6494093, 2.6954231

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9052828
time: 6.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9054662, upper bound: 0.9054656
time: 5.22 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7540431, 10.2373714, 7.7540421, 10.2373772, -2.0114293, 1.9989676
1: -19.2597008, -15.2714119, -19.2597084, -15.2714081, -2.4200568, 2.4333529
2: -6.5238419, -3.5489097, -6.5238442, -3.5489092, -1.9223185, 1.9225519
3: -10.8192215, -7.7928095, -10.8192244, -7.7928085, -2.3708744, 2.3794355
4: -13.5904942, -10.5921278, -13.5904999, -10.5921240, -2.1792388, 2.1938562
5: -4.6404009, -2.1593976, -4.6404033, -2.1593971, -1.7170377, 1.7191863
6: -4.5149145, -1.9159026, -4.5149150, -1.9158943, -2.0630388, 2.0158777
7: -12.8235540, -8.7824469, -12.8235579, -8.7824392, -2.9430232, 2.9241791
8: -5.4501781, -3.1462479, -5.4501796, -3.1462474, -1.4336684, 1.4131582
9: -1.9316421, 1.0465150, -1.9316492, 1.0465145, -2.6725225, 2.6946259

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9052827
time: 5.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054662, upper bound: 0.9094533
time: 5.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.73 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9028991, upper bound: 0.9022334
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9028991, upper bound: 0.9022334
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9028990, upper bound: 0.9022337
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9028991, upper bound: 0.9022335
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9089922
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9091752
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9052828
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9054662, upper bound: 0.9054656
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9052827
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.73
Output dim: 0, lower bound: -0.9054662, upper bound: 0.9094533

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7507610, 10.2255783, 7.7654247, 10.2054825, -2.0049582, 1.9976497
1: -19.2590218, -15.2659531, -19.2470665, -15.2779503, -2.4153142, 2.4346700
2: -6.5200481, -3.5485327, -6.5097742, -3.5558844, -1.9176331, 1.9061565
3: -10.8210812, -7.7849112, -10.8061295, -7.7989364, -2.3536286, 2.3625245
4: -13.5813208, -10.5878696, -13.5636244, -10.5980911, -2.1745090, 2.1846619
5: -4.6400499, -2.1589501, -4.6333523, -2.1689677, -1.7002540, 1.7079682
6: -4.5306988, -1.9217235, -4.5061011, -1.9474170, -2.0476999, 2.0078759
7: -12.8177814, -8.7821751, -12.7967453, -8.7944603, -2.9418979, 2.9130278
8: -5.4621334, -3.1508503, -5.4430361, -3.1705732, -1.4243848, 1.4060252
9: -1.9217916, 1.0568781, -1.8918252, 1.0419276, -2.6628494, 2.6692381

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9052838
time: 6.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9089918
time: 6.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7507610, 10.2255783, 7.7583351, 10.2116489, -2.0025768, 1.9973731
1: -19.2590218, -15.2659531, -19.2558765, -15.2687473, -2.4162045, 2.4347835
2: -6.5200481, -3.5485327, -6.5140500, -3.5513220, -1.9181056, 1.9073479
3: -10.8210812, -7.7849112, -10.8184986, -7.7879529, -2.3627219, 2.3709722
4: -13.5813208, -10.5878696, -13.5696211, -10.5921478, -2.1733975, 2.1823854
5: -4.6400499, -2.1589501, -4.6384292, -2.1632719, -1.7046628, 1.7060225
6: -4.5306988, -1.9217235, -4.5262289, -1.9295045, -2.0504408, 2.0017176
7: -12.8177814, -8.7821751, -12.8041096, -8.7873144, -2.9391489, 2.9228258
8: -5.4621334, -3.1508503, -5.4581299, -3.1575937, -1.4254880, 1.4029856
9: -1.9217916, 1.0568781, -1.9072652, 1.0527732, -2.6668205, 2.6775866

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9054670
time: 6.07 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9091753
time: 10.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 7.7540531, 10.2373638, 7.7486143, 10.2378464, -2.0080976, 2.0042262
1: -19.2596741, -15.2714119, -19.2621803, -15.2647104, -2.4266005, 2.4322886
2: -6.5238414, -3.5490210, -6.5266976, -3.5479188, -1.9210482, 1.9234867
3: -10.8192139, -7.7929287, -10.8222666, -7.7831626, -2.3807201, 2.3794451
4: -13.5902357, -10.5921268, -13.5909367, -10.5870676, -2.1811662, 2.1905894
5: -4.6402960, -2.1594000, -4.6407948, -2.1553016, -1.7148051, 1.7152038
6: -4.5149040, -1.9159192, -4.5329580, -1.9148694, -2.0534620, 2.0334148
7: -12.8235350, -8.7824535, -12.8300085, -8.7812052, -2.9379640, 2.9207649
8: -5.4501781, -3.1463552, -5.4638052, -3.1453629, -1.4271233, 1.4265287
9: -1.9316339, 1.0464828, -1.9343600, 1.0570168, -2.6832428, 2.6943541

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9056100, upper bound: 0.9055849
time: 6.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9056100, upper bound: 0.9094530
time: 5.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.56 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.56
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9052838
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9089918
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.56
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9054670
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 0, lower bound: -0.9015979, upper bound: 0.9091753
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.56
Output dim: 0, lower bound: -0.9056100, upper bound: 0.9055849
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 0, lower bound: -0.9056100, upper bound: 0.9094530

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 7.7488327, 10.2378445, 7.7654247, 10.2054825, -2.0068336, 1.9991817
1: -19.2621651, -15.2648096, -19.2470665, -15.2779503, -2.4190936, 2.4358292
2: -6.5261493, -3.5479281, -6.5097742, -3.5558844, -1.9237218, 1.9068279
3: -10.8218460, -7.7831702, -10.8061295, -7.7989364, -2.3533845, 2.3624473
4: -13.5909185, -10.5871487, -13.5636244, -10.5980911, -2.1860189, 2.1847558
5: -4.6407771, -2.1554253, -4.6333523, -2.1689677, -1.7001948, 1.7110538
6: -4.5329542, -1.9148774, -4.5061011, -1.9474170, -2.0474744, 2.0147049
7: -12.8299675, -8.7812080, -12.7967453, -8.7944603, -2.9546914, 2.9144793
8: -5.4638023, -3.1453819, -5.4430361, -3.1705732, -1.4245226, 1.4078228
9: -1.9343529, 1.0569463, -1.8918252, 1.0419276, -2.6755819, 2.6694078

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015738, upper bound: 0.9082149
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015736, upper bound: 0.9089674
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7488327, 10.2378445, 7.7583351, 10.2116489, -2.0045137, 2.0027640
1: -19.2621651, -15.2648096, -19.2558765, -15.2687473, -2.4199829, 2.4359422
2: -6.5261493, -3.5479281, -6.5140500, -3.5513220, -1.9242511, 1.9080169
3: -10.8218460, -7.7831702, -10.8184986, -7.7879529, -2.3624768, 2.3708940
4: -13.5909185, -10.5871487, -13.5696211, -10.5921478, -2.1849089, 2.1824803
5: -4.6407771, -2.1554253, -4.6384292, -2.1632719, -1.7045918, 1.7091076
6: -4.5329542, -1.9148774, -4.5262289, -1.9295045, -2.0522733, 2.0095758
7: -12.8299675, -8.7812080, -12.8041096, -8.7873144, -2.9519396, 2.9242449
8: -5.4638023, -3.1453819, -5.4581299, -3.1575937, -1.4274354, 1.4085958
9: -1.9343529, 1.0569463, -1.9072652, 1.0527732, -2.6795511, 2.6777549

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9018219, upper bound: 0.9083983
time: 5.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9018217, upper bound: 0.9091505
time: 8.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7486172, 10.2378454, 7.7486143, 10.2378464, -2.0091119, 1.9972606
1: -19.2621689, -15.2647152, -19.2621803, -15.2647104, -2.4204493, 2.4337487
2: -6.5261583, -3.5479281, -6.5266976, -3.5479188, -1.9212370, 1.9230609
3: -10.8222628, -7.7831693, -10.8222666, -7.7831626, -2.3801069, 2.3879900
4: -13.5909224, -10.5870695, -13.5909367, -10.5870676, -2.1768823, 2.1908255
5: -4.6407757, -2.1553025, -4.6407948, -2.1553016, -1.7181277, 1.7159491
6: -4.5329556, -1.9148768, -4.5329580, -1.9148694, -2.0552402, 2.0080781
7: -12.8300047, -8.7812080, -12.8300085, -8.7812052, -2.9392648, 2.9298191
8: -5.4638033, -3.1453681, -5.4638052, -3.1453629, -1.4297040, 1.4091892
9: -1.9343548, 1.0570135, -1.9343600, 1.0570168, -2.6787558, 2.7008538

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9019068, upper bound: 0.9086767
time: 5.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9019066, upper bound: 0.9094346
time: 6.20 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 32.96 seconds
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 32.96
Output dim: 0, lower bound: -0.9015738, upper bound: 0.9082149
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 32.96
Output dim: 0, lower bound: -0.9015736, upper bound: 0.9089674
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 32.96
Output dim: 0, lower bound: -0.9018219, upper bound: 0.9083983
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 32.96
Output dim: 0, lower bound: -0.9018217, upper bound: 0.9091505
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 32.96
Output dim: 0, lower bound: -0.9019068, upper bound: 0.9086767
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 32.96
Output dim: 0, lower bound: -0.9019066, upper bound: 0.9094346

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7488389, 10.2378416, 7.7600298, 10.2391396, -2.0096345, 2.0051081
1: -19.2621670, -15.2648125, -19.2761593, -15.2725048, -2.4261599, 2.4527538
2: -6.5261469, -3.5479295, -6.5134544, -3.5363698, -1.9434471, 1.9105895
3: -10.8218431, -7.7831702, -10.8157911, -7.7944107, -2.3638287, 2.3710904
4: -13.5909166, -10.5871496, -13.5976105, -10.5945940, -2.1910701, 2.2032909
5: -4.6407738, -2.1554275, -4.6380863, -2.1543217, -1.7248487, 1.7154312
6: -4.5329504, -1.9148771, -4.5125632, -1.9381943, -2.0519695, 2.0229473
7: -12.8299656, -8.7812128, -12.8065529, -8.7887011, -2.9637947, 2.9228139
8: -5.4638014, -3.1453857, -5.4595914, -3.1665254, -1.4285340, 1.4111192
9: -1.9343519, 1.0569456, -1.9168220, 1.0432250, -2.6861238, 2.6863308

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9008308, upper bound: 0.9089617
time: 7.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015687, upper bound: 0.9089625
time: 5.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 7.7488389, 10.2378416, 7.7529297, 10.2453089, -2.0133681, 2.0088663
1: -19.2621670, -15.2648125, -19.2849541, -15.2632723, -2.4270744, 2.4579475
2: -6.5261469, -3.5479295, -6.5177441, -3.5318589, -1.9439883, 1.9117849
3: -10.8218431, -7.7831702, -10.8281441, -7.7834320, -2.3729219, 2.3795710
4: -13.5909166, -10.5871496, -13.6036091, -10.5886364, -2.1899362, 2.2067876
5: -4.6407738, -2.1554275, -4.6431637, -2.1486318, -1.7292438, 1.7134902
6: -4.5329504, -1.9148771, -4.5327091, -1.9202664, -2.0592971, 2.0284166
7: -12.8299656, -8.7812128, -12.8139896, -8.7815790, -2.9695377, 2.9325919
8: -5.4638014, -3.1453857, -5.4746828, -3.1535640, -1.4316325, 1.4150643
9: -1.9343519, 1.0569456, -1.9322462, 1.0540683, -2.6915760, 2.6989241

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9010790, upper bound: 0.9091447
time: 7.31 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9018166, upper bound: 0.9091452
time: 6.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 7.7486172, 10.2378454, 7.7495346, 10.2377739, -2.0090261, 1.9963107
1: -19.2621689, -15.2647152, -19.2619705, -15.2654953, -2.4196358, 2.4335852
2: -6.5261583, -3.5479281, -6.5262680, -3.5479889, -1.9211483, 1.9225454
3: -10.8222628, -7.7831693, -10.8219748, -7.7832403, -2.3798046, 2.3874111
4: -13.5909224, -10.5870695, -13.5908604, -10.5879374, -2.1760235, 2.1907754
5: -4.6407757, -2.1553025, -4.6405010, -2.1555471, -1.7176337, 1.7153890
6: -4.5329556, -1.9148768, -4.5325818, -1.9149934, -2.0547819, 2.0073824
7: -12.8300047, -8.7812080, -12.8299236, -8.7815113, -2.9386120, 2.9293718
8: -5.4638033, -3.1453681, -5.4635620, -3.1456923, -1.4293747, 1.4089794
9: -1.9343548, 1.0570135, -1.9339280, 1.0567169, -2.6782026, 2.7002249

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013954, upper bound: 0.9086756
time: 5.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013954, upper bound: 0.9086761
time: 6.25 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 7.7486243, 10.2378426, 7.7432241, 10.2714109, -2.0220742, 2.0076218
1: -19.2621651, -15.2647181, -19.2909927, -15.2592030, -2.4275551, 2.4565897
2: -6.5261531, -3.5479269, -6.5303731, -3.5284758, -1.9409170, 1.9303696
3: -10.8222589, -7.7831697, -10.8317347, -7.7786860, -2.3931999, 2.3962183
4: -13.5909204, -10.5870714, -13.6248846, -10.5834675, -2.1820660, 2.2175524
5: -4.6407733, -2.1553054, -4.6455030, -2.1409135, -1.7431374, 1.7202740
6: -4.5329533, -1.9148774, -4.5395250, -1.9056467, -2.0638373, 2.0320158
7: -12.8300037, -8.7812128, -12.8397903, -8.7755566, -2.9585304, 2.9383883
8: -5.4638014, -3.1453714, -5.4801874, -3.1413774, -1.4338641, 1.4243927
9: -1.9343538, 1.0570102, -1.9591026, 1.0582826, -2.6929140, 2.7228565

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014119, upper bound: 0.9094288
time: 5.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9021495, upper bound: 0.9094297
time: 5.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 26.14 seconds
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9008308, upper bound: 0.9089617
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9015687, upper bound: 0.9089625
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9010790, upper bound: 0.9091447
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9018166, upper bound: 0.9091452
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9013954, upper bound: 0.9086756
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9013954, upper bound: 0.9086761
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9014119, upper bound: 0.9094288
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 26.14
Output dim: 0, lower bound: -0.9021495, upper bound: 0.9094297

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7505102, 10.2374115, 7.7600298, 10.2391396, -2.0070901, 2.0039718
1: -19.2591171, -15.2657757, -19.2761593, -15.2725048, -2.4229193, 2.4516988
2: -6.5255880, -3.5524454, -6.5134544, -3.5363698, -1.9428349, 1.9060605
3: -10.8212757, -7.7885904, -10.8157911, -7.7944107, -2.3632612, 2.3656673
4: -13.5855541, -10.5878410, -13.5976105, -10.5945940, -2.1857138, 2.2026947
5: -4.6377769, -2.1565018, -4.6380863, -2.1543217, -1.7218919, 1.7143543
6: -4.5297279, -1.9153399, -4.5125632, -1.9381943, -2.0480657, 2.0209570
7: -12.8289003, -8.7880230, -12.8065529, -8.7887011, -2.9619246, 2.9159517
8: -5.4629984, -3.1465588, -5.4595914, -3.1665254, -1.4268770, 1.4089122
9: -1.9307089, 1.0549207, -1.9168220, 1.0432250, -2.6825094, 2.6843009

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9008296, upper bound: 0.9082226
time: 7.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9008296, upper bound: 0.9089617
time: 7.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7341275, 10.2535610, 7.7600398, 10.2391396, -2.0225539, 2.0098262
1: -19.2720604, -15.2551603, -19.2761459, -15.2725048, -2.4346027, 2.4551485
2: -6.5672455, -3.5410604, -6.5134549, -3.5363860, -1.9556127, 1.9192007
3: -10.8813572, -7.7734151, -10.8157873, -7.7944398, -2.3797793, 2.3831639
4: -13.5971956, -10.5196257, -13.5975981, -10.5945978, -2.2006326, 2.2049956
5: -4.6446581, -2.1232872, -4.6380739, -2.1543264, -1.7292862, 1.7389982
6: -4.5398989, -1.8969560, -4.5125494, -1.9381946, -2.0597014, 2.0284982
7: -12.9026451, -8.7715273, -12.8065472, -8.7887383, -2.9684601, 2.9395275
8: -5.4698434, -3.1403890, -5.4595890, -3.1665292, -1.4359498, 1.4149255
9: -1.9521933, 1.0577266, -1.9168105, 1.0432129, -2.7037621, 2.6874647

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9015689, upper bound: 0.9082225
time: 5.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9015689, upper bound: 0.9089616
time: 6.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7505102, 10.2374115, 7.7529297, 10.2453089, -2.0108223, 2.0077298
1: -19.2591171, -15.2657757, -19.2849541, -15.2632723, -2.4238338, 2.4568925
2: -6.5255880, -3.5524454, -6.5177441, -3.5318589, -1.9433589, 1.9072492
3: -10.8212757, -7.7885904, -10.8281441, -7.7834320, -2.3723545, 2.3741479
4: -13.5855541, -10.5878410, -13.6036091, -10.5886364, -2.1845798, 2.2061903
5: -4.6377769, -2.1565018, -4.6431637, -2.1486318, -1.7263012, 1.7124128
6: -4.5297279, -1.9153399, -4.5327091, -1.9202664, -2.0553932, 2.0264251
7: -12.8289003, -8.7880230, -12.8139896, -8.7815790, -2.9676685, 2.9257379
8: -5.4629984, -3.1465588, -5.4746828, -3.1535640, -1.4299803, 1.4128580
9: -1.9307089, 1.0549207, -1.9322462, 1.0540683, -2.6879597, 2.6968937

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9010784, upper bound: 0.9089340
time: 6.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9010784, upper bound: 0.9091444
time: 7.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7341275, 10.2535610, 7.7529407, 10.2453070, -2.0262818, 2.0135827
1: -19.2720604, -15.2551603, -19.2849426, -15.2632761, -2.4355173, 2.4603405
2: -6.5672455, -3.5410604, -6.5177426, -3.5318708, -1.9595985, 1.9204695
3: -10.8813572, -7.7734151, -10.8281393, -7.7834640, -2.3874054, 2.3916454
4: -13.5971956, -10.5196257, -13.6035957, -10.5886374, -2.1994991, 2.2084875
5: -4.6446581, -2.1232872, -4.6431541, -2.1486361, -1.7337184, 1.7433655
6: -4.5398989, -1.8969560, -4.5326958, -1.9202693, -2.0671501, 2.0334027
7: -12.9026451, -8.7715273, -12.8139877, -8.7816181, -2.9741983, 2.9494934
8: -5.4698434, -3.1403890, -5.4746799, -3.1535673, -1.4414961, 1.4188662
9: -1.9521933, 1.0577266, -1.9322319, 1.0540581, -2.7091904, 2.7000561

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9018167, upper bound: 0.9084061
time: 5.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9018167, upper bound: 0.9091452
time: 6.24 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 33.50 seconds
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9008296, upper bound: 0.9082226
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9008296, upper bound: 0.9089617
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9015689, upper bound: 0.9082225
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9015689, upper bound: 0.9089616
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9010784, upper bound: 0.9089340
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9010784, upper bound: 0.9091444
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9018167, upper bound: 0.9084061
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 33.50
Output dim: 0, lower bound: -0.9018167, upper bound: 0.9091452
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 33.50
Output dim: 0, lower bound: -0.9013954, upper bound: 0.9086756
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 33.50
Output dim: 0, lower bound: -0.9013954, upper bound: 0.9086761
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 33.50
Output dim: 0, lower bound: -0.9014119, upper bound: 0.9094288
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 33.50
Output dim: 0, lower bound: -0.9021495, upper bound: 0.9094297
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.0339293479919434
rel_dist={0: [-0.9094671207244254, 0.9094665870331546]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2426.23 seconds
