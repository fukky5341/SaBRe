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
execution time: IAR + LP analysis = 14.31 + 35.58 = 49.89 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3550.11 seconds, max iter: 100)

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
Binary search time: 210.22 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3339.89 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271530, upper bound: 1.4199076
time: 5.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297444, upper bound: 1.4297441
time: 8.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 0, lower bound: -1.4271530, upper bound: 1.4199076
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.63
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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271333, upper bound: 1.4111447
time: 6.32 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4271332, upper bound: 1.4198879
time: 6.23 seconds

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4199074, upper bound: 1.4271532
time: 6.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4199075, upper bound: 1.4297449
time: 6.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.45 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 27.45
Output dim: 0, lower bound: -1.4271333, upper bound: 1.4111447
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 27.45
Output dim: 0, lower bound: -1.4271332, upper bound: 1.4198879
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.45
Output dim: 0, lower bound: -1.4199074, upper bound: 1.4271532
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.45
Output dim: 0, lower bound: -1.4199075, upper bound: 1.4297449

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7572622, 10.2258034, -2.3390517, 2.3248987
1: -19.2184887, -15.2969704, -19.2492027, -15.2753029, -2.9575901, 2.9634423
2: -6.4989195, -3.5554147, -6.5178347, -3.5500522, -2.3225288, 2.3457212
3: -10.7896709, -7.8088303, -10.8121061, -7.7944794, -2.7581701, 2.7601304
4: -13.5483303, -10.6108570, -13.5801487, -10.5945702, -2.6679525, 2.6824851
5: -4.6248550, -2.1774464, -4.6363554, -2.1629639, -2.0529733, 2.0480394
6: -4.4857645, -1.9801078, -4.5127292, -1.9333184, -2.3398347, 2.3254337
7: -12.7819271, -8.8199348, -12.8161373, -8.7911253, -3.4699087, 3.4770966
8: -5.4283876, -3.1802011, -5.4474945, -3.1549516, -1.7555041, 1.7551730
9: -1.8567619, 1.0274441, -1.9126530, 1.0461040, -2.9028659, 2.9400971

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4111448
time: 5.00 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198877, upper bound: 1.4111449
time: 5.48 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7562246, 10.2317152, -2.3550043, 2.3345003
1: -19.2248154, -15.2928429, -19.2507172, -15.2747021, -2.9599476, 2.9695239
2: -6.5115700, -3.5520153, -6.5208359, -3.5497639, -2.3391027, 2.3522635
3: -10.7933302, -7.8039842, -10.8126593, -7.7936406, -2.7630353, 2.7793455
4: -13.5695877, -10.6057549, -13.5847692, -10.5941849, -2.6800613, 2.6928563
5: -4.6272178, -2.1693721, -4.6367040, -2.1612105, -2.0616055, 2.0574441
6: -4.4925199, -1.9654692, -4.5138092, -1.9300066, -2.3509469, 2.3309355
7: -12.8078947, -8.8138428, -12.8221092, -8.7906666, -3.4807734, 3.4900208
8: -5.4340749, -3.1679506, -5.4483004, -3.1522942, -1.7692003, 1.7602539
9: -1.8838563, 1.0316887, -1.9187107, 1.0461683, -2.9300246, 2.9503994

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4198878
time: 5.24 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4198878
time: 5.32 seconds

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4271332
time: 6.35 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198874, upper bound: 1.4271332
time: 6.34 seconds

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

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4209822
time: 5.69 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4297243
time: 5.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.23 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4111448
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198877, upper bound: 1.4111449
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4198878
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4198878
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4271332
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198874, upper bound: 1.4271332
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4209822
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4297243

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7694950, 10.2115049, -2.3244605, 2.3125737
1: -19.2184887, -15.2969704, -19.2232952, -15.2934570, -2.9392700, 2.9380789
2: -6.4989195, -3.5554147, -6.5085816, -3.5523045, -2.3200684, 2.3354394
3: -10.7896709, -7.8088303, -10.7927866, -7.8048296, -2.7449198, 2.7438264
4: -13.5483303, -10.6108570, -13.5649776, -10.6061420, -2.6566725, 2.6672120
5: -4.6248550, -2.1774464, -4.6268711, -2.1711395, -2.0440450, 2.0388765
6: -4.4857645, -1.9801078, -4.4914341, -1.9687774, -2.3043051, 2.3039184
7: -12.7819271, -8.8199348, -12.8019123, -8.8142977, -3.4490371, 3.4623599
8: -5.4283876, -3.1802011, -5.4332557, -3.1706104, -1.7399170, 1.7406049
9: -1.8567619, 1.0274441, -1.8778095, 1.0316250, -2.8883870, 2.9052536

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4188539, upper bound: 1.4111400
time: 5.34 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198825, upper bound: 1.4111399
time: 5.49 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7550731, 10.2314625, -2.3432245, 2.3272166
1: -19.2184887, -15.2969704, -19.2581863, -15.2720079, -2.9608331, 2.9723663
2: -6.4989195, -3.5554147, -6.5208402, -3.5491958, -2.3239241, 2.3497026
3: -10.7896709, -7.8088303, -10.8186674, -7.7936425, -2.7563553, 2.7684364
4: -13.5483303, -10.6108570, -13.5858727, -10.5925064, -2.6692657, 2.6880431
5: -4.6248550, -2.1774464, -4.6400533, -2.1611423, -2.0532184, 2.0519667
6: -4.4857645, -1.9801078, -4.5138359, -1.9192102, -2.3505607, 2.3268127
7: -12.7819271, -8.8199348, -12.8175869, -8.7829056, -3.4795423, 3.4759007
8: -5.4283876, -3.1802011, -5.4493790, -3.1489077, -1.7615547, 1.7573352
9: -1.8567619, 1.0274441, -1.9255891, 1.0464511, -2.9032130, 2.9530332

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4188541, upper bound: 1.4111399
time: 5.92 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198827, upper bound: 1.4111400
time: 6.06 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7684388, 10.2174206, -2.3404117, 2.3221931
1: -19.2248154, -15.2928429, -19.2248173, -15.2928391, -2.9415941, 2.9441438
2: -6.5115700, -3.5520153, -6.5115714, -3.5520163, -2.3366375, 2.3419671
3: -10.7933302, -7.8039842, -10.7933292, -7.8039856, -2.7497916, 2.7630773
4: -13.5695877, -10.6057549, -13.5695906, -10.6057529, -2.6687717, 2.6775804
5: -4.6272178, -2.1693721, -4.6272192, -2.1693683, -2.0526466, 2.0482774
6: -4.4925199, -1.9654692, -4.4925184, -1.9654677, -2.3154149, 2.3094053
7: -12.8078947, -8.8138428, -12.8079023, -8.8138399, -3.4598823, 3.4752803
8: -5.4340749, -3.1679506, -5.4340787, -3.1679492, -1.7536077, 1.7456937
9: -1.8838563, 1.0316887, -1.8838596, 1.0316887, -2.9155450, 2.9155483

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198875
time: 6.90 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198880
time: 6.53 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7540426, 10.2373781, -2.3604569, 2.3368127
1: -19.2248154, -15.2928429, -19.2597046, -15.2714100, -2.9632044, 2.9784455
2: -6.5115700, -3.5520153, -6.5238457, -3.5489109, -2.3405027, 2.3562484
3: -10.7933302, -7.8039842, -10.8192225, -7.7928071, -2.7612233, 2.7876620
4: -13.5695877, -10.6057549, -13.5904970, -10.5921268, -2.6813774, 2.6984148
5: -4.6272178, -2.1693721, -4.6404028, -2.1593966, -2.0618629, 2.0613699
6: -4.4925199, -1.9654692, -4.5149150, -1.9159002, -2.3646674, 2.3323288
7: -12.8078947, -8.8138428, -12.8235588, -8.7824440, -3.4903517, 3.4888220
8: -5.4340749, -3.1679506, -5.4501791, -3.1462469, -1.7752509, 1.7624121
9: -1.8838563, 1.0316887, -1.9316473, 1.0465150, -2.9303713, 2.9633360

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111445, upper bound: 1.4198878
time: 6.28 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111444, upper bound: 1.4198880
time: 5.75 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 7.7550731, 10.2314625, 7.7782664, 10.1912088, -2.3272166, 2.3432248
1: -19.2581863, -15.2720079, -19.2184887, -15.2969704, -2.9723663, 2.9608340
2: -6.5208402, -3.5491958, -6.4989195, -3.5554147, -2.3497033, 2.3239241
3: -10.8186674, -7.7936425, -10.7896709, -7.8088303, -2.7684374, 2.7563558
4: -13.5858727, -10.5925064, -13.5483303, -10.6108570, -2.6880431, 2.6692653
5: -4.6400533, -2.1611423, -4.6248550, -2.1774464, -2.0519667, 2.0532181
6: -4.5138359, -1.9192102, -4.4857645, -1.9801078, -2.3268127, 2.3505607
7: -12.8175869, -8.7829056, -12.7819271, -8.8199348, -3.4759007, 3.4795423
8: -5.4493790, -3.1489077, -5.4283876, -3.1802011, -1.7573349, 1.7615547
9: -1.9255891, 1.0464511, -1.8567619, 1.0274441, -2.9530332, 2.9032130

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4065536, upper bound: 1.4248968
time: 7.07 seconds

## Relational analysis of IS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111393, upper bound: 1.4261001
time: 6.06 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111414, upper bound: 1.4271278
time: 6.76 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 7.7540426, 10.2373781, 7.7684426, 10.2174129, -2.3368125, 2.3604569
1: -19.2597046, -15.2714100, -19.2248154, -15.2928429, -2.9784460, 2.9632049
2: -6.5238457, -3.5489109, -6.5115700, -3.5520153, -2.3562484, 2.3405025
3: -10.8192225, -7.7928071, -10.7933302, -7.8039842, -2.7876625, 2.7612224
4: -13.5904970, -10.5921268, -13.5695877, -10.6057549, -2.6984148, 2.6813779
5: -4.6404028, -2.1593966, -4.6272178, -2.1693721, -2.0613699, 2.0618629
6: -4.5149150, -1.9159002, -4.4925199, -1.9654692, -2.3323283, 2.3646677
7: -12.8235588, -8.7824440, -12.8078947, -8.8138428, -3.4888220, 3.4903517
8: -5.4501791, -3.1462469, -5.4340749, -3.1679506, -1.7624121, 1.7752509
9: -1.9316473, 1.0465150, -1.8838563, 1.0316887, -2.9633360, 2.9303713

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4152951, upper bound: 1.4248969
time: 6.13 seconds

## Relational analysis of IS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198877, upper bound: 1.4183917
time: 6.09 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4271333
time: 6.21 seconds

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

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4196500, upper bound: 1.4197661
time: 5.54 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4219926, upper bound: 1.4209761
time: 5.49 seconds

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

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4196500, upper bound: 1.4285014
time: 6.13 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4219926, upper bound: 1.4297201
time: 6.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.57 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4188539, upper bound: 1.4111400
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4198825, upper bound: 1.4111399
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4188541, upper bound: 1.4111399
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4198827, upper bound: 1.4111400
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198875
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4111443, upper bound: 1.4198880
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4111445, upper bound: 1.4198878
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4111444, upper bound: 1.4198880
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4111393, upper bound: 1.4261001
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4111414, upper bound: 1.4271278
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4198877, upper bound: 1.4183917
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4198876, upper bound: 1.4271333
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4196500, upper bound: 1.4197661
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4219926, upper bound: 1.4209761
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4196500, upper bound: 1.4285014
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.57
Output dim: 0, lower bound: -1.4219926, upper bound: 1.4297201

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 7.7806282, 10.1899796, 7.7694950, 10.2115049, -2.3185883, 2.3082986
1: -19.2077351, -15.2979126, -19.2232952, -15.2934570, -2.9280958, 2.9370518
2: -6.4978967, -3.5649998, -6.5085816, -3.5523045, -2.3192139, 2.3258593
3: -10.7870522, -7.8204222, -10.7927866, -7.8048296, -2.7425451, 2.7324257
4: -13.5415668, -10.6120348, -13.5649776, -10.6061420, -2.6498842, 2.6664066
5: -4.6162295, -2.1803131, -4.6268711, -2.1711395, -2.0353789, 2.0359275
6: -4.4798203, -1.9806833, -4.4914341, -1.9687774, -2.2954416, 2.2997398
7: -12.7794056, -8.8369770, -12.8019123, -8.8142977, -3.4444776, 3.4447842
8: -5.4262915, -3.1817360, -5.4332557, -3.1706104, -1.7355230, 1.7360129
9: -1.8517418, 1.0188844, -1.8778095, 1.0316250, -2.8833668, 2.8966939

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4101120, upper bound: 1.4111401
time: 5.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4101120, upper bound: 1.4111395
time: 5.41 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 7.7653236, 10.1997185, 7.7694969, 10.2115040, -2.3330512, 2.3355865
1: -19.2253227, -15.2469673, -19.2232933, -15.2934589, -2.9486914, 2.9811490
2: -6.5393338, -3.5509324, -6.5085812, -3.5523088, -2.3621697, 2.3437805
3: -10.8433599, -7.8034406, -10.7927856, -7.8048377, -2.7854910, 2.7530642
4: -13.5508528, -10.5742722, -13.5649767, -10.6061420, -2.6617384, 2.6991034
5: -4.6317458, -2.1358287, -4.6268649, -2.1711426, -2.0528264, 2.0666804
6: -4.4913344, -1.9598174, -4.4914322, -1.9687779, -2.3323760, 2.3204064
7: -12.8587818, -8.8074350, -12.8019085, -8.8143044, -3.4937878, 3.4940367
8: -5.4382496, -3.1745210, -5.4332571, -3.1706123, -1.7652261, 1.7432609
9: -1.9152765, 1.0279934, -1.8778048, 1.0316205, -2.9468970, 2.9057982

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111412, upper bound: 1.4111399
time: 5.56 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111412, upper bound: 1.4111395
time: 6.97 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 7.7806282, 10.1899796, 7.7550731, 10.2314625, -2.3373384, 2.3229418
1: -19.2077351, -15.2979126, -19.2581863, -15.2720079, -2.9496598, 2.9713397
2: -6.4978967, -3.5649998, -6.5208402, -3.5491958, -2.3230696, 2.3401234
3: -10.7870522, -7.8204222, -10.8186674, -7.7936425, -2.7539806, 2.7570357
4: -13.5415668, -10.6120348, -13.5858727, -10.5925064, -2.6624756, 2.6872373
5: -4.6162295, -2.1803131, -4.6400533, -2.1611423, -2.0445514, 2.0490174
6: -4.4798203, -1.9806833, -4.5138359, -1.9192102, -2.3417158, 2.3226347
7: -12.7794056, -8.8369770, -12.8175869, -8.7829056, -3.4749827, 3.4583249
8: -5.4262915, -3.1817360, -5.4493790, -3.1489077, -1.7571607, 1.7527430
9: -1.8517418, 1.0188844, -1.9255891, 1.0464511, -2.8981929, 2.9444735

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4238647, upper bound: 1.4065488
time: 7.15 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4173618, upper bound: 1.4111414
time: 7.61 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4173599, upper bound: 1.4111384
time: 6.80 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 7.7653236, 10.1997185, 7.7550750, 10.2314615, -2.3517609, 2.3463593
1: -19.2253227, -15.2469673, -19.2581806, -15.2720070, -2.9702539, 3.0021768
2: -6.5393338, -3.5509324, -6.5208397, -3.5492029, -2.3660259, 2.3580449
3: -10.8433599, -7.8034406, -10.8186665, -7.7936492, -2.7905054, 2.7776752
4: -13.5508528, -10.5742722, -13.5858698, -10.5925074, -2.6743307, 2.7115712
5: -4.6317458, -2.1358287, -4.6400485, -2.1611454, -2.0619984, 2.0761352
6: -4.4913344, -1.9598174, -4.5138350, -1.9192117, -2.3606594, 2.3402393
7: -12.8587818, -8.8074350, -12.8175850, -8.7829113, -3.5134916, 3.5075788
8: -5.4382496, -3.1745210, -5.4493766, -3.1489105, -1.7763708, 1.7599914
9: -1.9152765, 1.0279934, -1.9255862, 1.0464463, -2.9605484, 2.9535797

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4248919, upper bound: 1.4065488
time: 6.29 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4183881, upper bound: 1.4111398
time: 6.03 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4183881, upper bound: 1.4111398
time: 5.60 seconds

## BFS IS instance: IS_A1_A2_B1_B1

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

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111393, upper bound: 1.4188535
time: 6.31 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111393, upper bound: 1.4198821
time: 7.25 seconds

## BFS IS instance: IS_A1_A2_B1_B2

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

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4101100, upper bound: 1.4198833
time: 6.69 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4111392, upper bound: 1.4198831
time: 5.48 seconds

## BFS IS instance: IS_A1_A2_B2_B1

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.3750882148742676
rel_dist={0: [-1.4297502739106367, 1.4297521247570089]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565350, upper bound: 1.0523701
time: 7.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573862, upper bound: 1.0573857
time: 10.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.58
Output dim: 0, lower bound: -1.0565350, upper bound: 1.0523701
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.58
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

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0562603, upper bound: 1.0471093
time: 6.83 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565244, upper bound: 1.0523589
time: 6.51 seconds

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0571197, upper bound: 1.0521420
time: 5.73 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573755, upper bound: 1.0573744
time: 5.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.44 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 26.44
Output dim: 0, lower bound: -1.0562603, upper bound: 1.0471093
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 26.44
Output dim: 0, lower bound: -1.0565244, upper bound: 1.0523589
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 26.44
Output dim: 0, lower bound: -1.0571197, upper bound: 1.0521420
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 26.44
Output dim: 0, lower bound: -1.0573755, upper bound: 1.0573744

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 7.7782664, 10.1912088, 7.7594471, 10.2179432, -2.0754342, 2.0666857
1: -19.2184887, -15.2969704, -19.2422504, -15.2779140, -2.5349317, 2.5341420
2: -6.4989195, -3.5554147, -6.5138125, -3.5508311, -1.9979978, 2.0176511
3: -10.7896709, -7.8088303, -10.8074064, -7.7956238, -2.4363537, 2.4338775
4: -13.5483303, -10.6108570, -13.5731440, -10.5962143, -2.2881517, 2.2967629
5: -4.6248550, -2.1774464, -4.6336899, -2.1654165, -1.7840567, 1.7788596
6: -4.4857645, -1.9801078, -4.5112262, -1.9449041, -2.0907178, 2.0859289
7: -12.7819271, -8.8199348, -12.8110619, -8.7968330, -3.0458755, 3.0544386
8: -5.4283876, -3.1802011, -5.4456720, -3.1607761, -1.4886587, 1.4916959
9: -1.8567619, 1.0274441, -1.8999076, 1.0458264, -2.6936884, 2.7190809

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0558426, upper bound: 1.0471075
time: 6.88 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0562600, upper bound: 1.0471072
time: 7.93 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 7.7684426, 10.2174129, 7.7576704, 10.2280102, -2.0955725, 2.0738969
1: -19.2248154, -15.2928429, -19.2448311, -15.2768841, -2.5346556, 2.5414844
2: -6.5115700, -3.5520153, -6.5188708, -3.5503392, -2.0138187, 2.0262456
3: -10.7933302, -7.8039842, -10.8083515, -7.7941904, -2.4418473, 2.4517145
4: -13.5695877, -10.6057549, -13.5810127, -10.5955544, -2.2974606, 2.3110332
5: -4.6272178, -2.1693721, -4.6342835, -2.1624115, -1.7934661, 1.7885303
6: -4.4925199, -1.9654692, -4.5130796, -1.9392797, -2.1044836, 2.0907421
7: -12.8078947, -8.8138428, -12.8211451, -8.7960434, -3.0537672, 3.0717196
8: -5.4340749, -3.1679506, -5.4470539, -3.1562667, -1.5038300, 1.4960406
9: -1.8838563, 1.0316887, -1.9102111, 1.0459359, -2.7170458, 2.7329159

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512839, upper bound: 1.0520863
time: 6.55 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512844, upper bound: 1.0523575
time: 9.12 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 7.7637668, 10.2111721, 7.7557983, 10.2273159, -2.0991793, 2.0787072
1: -19.2533894, -15.2754498, -19.2571297, -15.2724237, -2.5619860, 2.5705471
2: -6.5111165, -3.5523076, -6.5187750, -3.5493996, -2.0145240, 2.0284169
3: -10.8155174, -7.7976027, -10.8182793, -7.7942338, -2.4614096, 2.4693685
4: -13.5691833, -10.5972061, -13.5826302, -10.5927801, -2.2978230, 2.3185306
5: -4.6380377, -2.1673660, -4.6398096, -2.1623816, -1.7969537, 1.7946870
6: -4.5081902, -1.9305443, -4.5130706, -1.9215159, -2.1371703, 2.0930381
7: -12.7976437, -8.7885475, -12.8134766, -8.7832270, -3.0753727, 3.0670595
8: -5.4444995, -3.1584921, -5.4488101, -3.1507549, -1.5154731, 1.4973977
9: -1.9045305, 1.0422702, -1.9213433, 1.0464058, -2.7215986, 2.7562633

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514485, upper bound: 1.0516121
time: 7.87 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0571157, upper bound: 1.0521382
time: 6.64 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 7.7540431, 10.2373714, 7.7540388, 10.2373838, -2.1192122, 2.0859485
1: -19.2597008, -15.2714119, -19.2597103, -15.2714062, -2.5617352, 2.5778065
2: -6.5238419, -3.5489097, -6.5238481, -3.5489058, -2.0304360, 2.0370188
3: -10.8192215, -7.7928095, -10.8192282, -7.7928076, -2.4669709, 2.4871945
4: -13.5904942, -10.5921278, -13.5905037, -10.5921240, -2.3071613, 2.3328357
5: -4.6404009, -2.1593976, -4.6404047, -2.1593952, -1.8064241, 1.8042066
6: -4.5149145, -1.9159026, -4.5149159, -1.9158906, -2.1509128, 2.0978856
7: -12.8235540, -8.7824469, -12.8235607, -8.7824383, -3.0832510, 3.0843110
8: -5.4501781, -3.1462479, -5.4501820, -3.1462431, -1.5305805, 1.5017338
9: -1.9316421, 1.0465150, -1.9316545, 1.0465150, -2.7450390, 2.7701054

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517138, upper bound: 1.0568508
time: 8.19 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573715, upper bound: 1.0573708
time: 6.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.74 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0558426, upper bound: 1.0471075
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0562600, upper bound: 1.0471072
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0512839, upper bound: 1.0520863
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0512844, upper bound: 1.0523575
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0514485, upper bound: 1.0516121
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0571157, upper bound: 1.0521382
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0517138, upper bound: 1.0568508
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.74
Output dim: 0, lower bound: -1.0573715, upper bound: 1.0573708

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: 7.7806282, 10.1899796, 7.7594471, 10.2179432, -2.0695624, 2.0624108
1: -19.2077351, -15.2979126, -19.2422504, -15.2779140, -2.5237575, 2.5331149
2: -6.4978967, -3.5649998, -6.5138125, -3.5508311, -1.9971442, 2.0080714
3: -10.7870522, -7.8204222, -10.8074064, -7.7956238, -2.4339790, 2.4224768
4: -13.5415668, -10.6120348, -13.5731440, -10.5962143, -2.2813635, 2.2959576
5: -4.6162295, -2.1803131, -4.6336899, -2.1654165, -1.7753901, 1.7759106
6: -4.4798203, -1.9806833, -4.5112262, -1.9449041, -2.0818543, 2.0817499
7: -12.7794056, -8.8369770, -12.8110619, -8.7968330, -3.0413160, 3.0368633
8: -5.4262915, -3.1817360, -5.4456720, -3.1607761, -1.4842646, 1.4871037
9: -1.8517418, 1.0188844, -1.8999076, 1.0458264, -2.6895266, 2.7105174

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523190, upper bound: 1.0471076
time: 6.65 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523190, upper bound: 1.0471092
time: 5.87 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: 7.7653236, 10.1997185, 7.7594509, 10.2179413, -2.0840216, 2.0852675
1: -19.2253227, -15.2469673, -19.2422447, -15.2779160, -2.5405636, 2.5674663
2: -6.5393338, -3.5509324, -6.5138078, -3.5508387, -2.0400972, 2.0222631
3: -10.8433599, -7.8034406, -10.8073997, -7.7956328, -2.4714580, 2.4396825
4: -13.5508528, -10.5742722, -13.5731411, -10.5962133, -2.2906466, 2.3207185
5: -4.6317458, -2.1358287, -4.6336823, -2.1654229, -1.7901154, 1.8034647
6: -4.4913344, -1.9598174, -4.5112214, -1.9449058, -2.1036031, 2.1024084
7: -12.8587818, -8.8074350, -12.8110600, -8.7968426, -3.0829468, 3.0738211
8: -5.4382496, -3.1745210, -5.4456687, -3.1607780, -1.5043173, 1.4943476
9: -1.9152765, 1.0279934, -1.8999014, 1.0458183, -2.7540379, 2.7197123

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527298, upper bound: 1.0471075
time: 5.64 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527298, upper bound: 1.0471073
time: 5.67 seconds

## BFS IS instance: IS_A1_A2_B1

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

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512818, upper bound: 1.0516750
time: 6.07 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512818, upper bound: 1.0520853
time: 5.81 seconds

## BFS IS instance: IS_A1_A2_B2

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508640, upper bound: 1.0523557
time: 7.10 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512817, upper bound: 1.0523575
time: 5.54 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 7.7645831, 10.2083635, 7.7574439, 10.2216196, -2.0925574, 2.0737228
1: -19.2502689, -15.2766790, -19.2507992, -15.2749157, -2.5560203, 2.5628338
2: -6.5104556, -3.5551784, -6.5174279, -3.5529699, -2.0100994, 2.0252905
3: -10.8108883, -7.7982597, -10.8088923, -7.7955585, -2.4550462, 2.4588752
4: -13.5664406, -10.5976439, -13.5770607, -10.5936613, -2.2931776, 2.3122649
5: -4.6357269, -2.1681561, -4.6351213, -2.1639667, -1.7911096, 1.7883637
6: -4.5071616, -1.9388648, -4.5109868, -1.9383961, -2.1189585, 2.0820441
7: -12.7971992, -8.7914639, -12.8125744, -8.7891436, -3.0675983, 3.0639133
8: -5.4437790, -3.1644440, -5.4473467, -3.1628404, -1.5025272, 1.4897492
9: -1.8982663, 1.0421023, -1.9086261, 1.0460651, -2.7149010, 2.7432427

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510298, upper bound: 1.0516098
time: 7.34 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514464, upper bound: 1.0516101
time: 5.61 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 7.7637768, 10.2111673, 7.7503700, 10.2277880, -2.0963449, 2.0840163
1: -19.2533684, -15.2754517, -19.2595978, -15.2657290, -2.5685377, 2.5698547
2: -6.5111170, -3.5523996, -6.5218048, -3.5484066, -2.0136757, 2.0296381
3: -10.8155088, -7.7977028, -10.8212976, -7.7845845, -2.4712586, 2.4693532
4: -13.5689621, -10.5972061, -13.5830669, -10.5877209, -2.2999096, 2.3157496
5: -4.6379490, -2.1673684, -4.6402035, -2.1582849, -1.7950306, 1.7912116
6: -4.5081820, -1.9305604, -4.5311127, -1.9204891, -2.1288180, 2.1105866
7: -12.7976265, -8.7885542, -12.8199387, -8.7819929, -3.0709295, 3.0638757
8: -5.4445009, -3.1585808, -5.4624376, -3.1498661, -1.5097873, 1.5107813
9: -1.9045210, 1.0422418, -1.9240575, 1.0569069, -2.7323217, 2.7563434

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0566954, upper bound: 1.0521361
time: 7.13 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0571136, upper bound: 1.0521362
time: 6.41 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 7.7548494, 10.2345638, 7.7556772, 10.2316856, -2.1126437, 2.0809231
1: -19.2565804, -15.2726326, -19.2533817, -15.2738972, -2.5557613, 2.5701008
2: -6.5231800, -3.5517776, -6.5225024, -3.5524809, -2.0260029, 2.0338960
3: -10.8145962, -7.7934589, -10.8098412, -7.7941289, -2.4606109, 2.4766979
4: -13.5877495, -10.5925608, -13.5849352, -10.5930042, -2.3025169, 2.3265767
5: -4.6380911, -2.1601739, -4.6357179, -2.1609693, -1.8005571, 1.7978942
6: -4.5138917, -1.9242227, -4.5128365, -1.9327689, -2.1327062, 2.0868793
7: -12.8231115, -8.7853622, -12.8226604, -8.7883549, -3.0755272, 3.0811653
8: -5.4494581, -3.1522021, -5.4487200, -3.1583290, -1.5176413, 1.4940906
9: -1.9253755, 1.0463471, -1.9189339, 1.0461740, -2.7383356, 2.7570782

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512918, upper bound: 1.0568489
time: 13.60 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517118, upper bound: 1.0568488
time: 7.34 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 7.7540541, 10.2373676, 7.7486162, 10.2378540, -2.1163726, 2.0912080
1: -19.2596722, -15.2714119, -19.2621841, -15.2647114, -2.5682836, 2.5771322
2: -6.5238404, -3.5490017, -6.5268464, -3.5479136, -2.0295777, 2.0382071
3: -10.8192139, -7.7929106, -10.8222685, -7.7831602, -2.4768200, 2.4871097
4: -13.5902748, -10.5921268, -13.5909414, -10.5870657, -2.3092303, 2.3300538
5: -4.6403122, -2.1594009, -4.6407995, -2.1552987, -1.8044462, 1.8007288
6: -4.5149059, -1.9159163, -4.5329599, -1.9148625, -2.1425619, 2.1154261
7: -12.8235416, -8.7824507, -12.8300142, -8.7812023, -3.0788069, 3.0811701
8: -5.4501758, -3.1463375, -5.4638038, -3.1453590, -1.5249021, 1.5151176
9: -1.9316349, 1.0464876, -1.9343662, 1.0570199, -2.7557611, 2.7701788

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0569435, upper bound: 1.0573690
time: 6.60 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573694, upper bound: 1.0573688
time: 5.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.79 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0523190, upper bound: 1.0471076
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0523190, upper bound: 1.0471092
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0527298, upper bound: 1.0471075
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0527298, upper bound: 1.0471073
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0512818, upper bound: 1.0516750
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0512818, upper bound: 1.0520853
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0508640, upper bound: 1.0523557
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0512817, upper bound: 1.0523575
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0510298, upper bound: 1.0516098
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0514464, upper bound: 1.0516101
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0566954, upper bound: 1.0521361
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0571136, upper bound: 1.0521362
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0512918, upper bound: 1.0568489
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0517118, upper bound: 1.0568488
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0569435, upper bound: 1.0573690
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.79
Output dim: 0, lower bound: -1.0573694, upper bound: 1.0573688

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 7.7806282, 10.1899796, 7.7702370, 10.2073498, -2.0585322, 2.0516095
1: -19.2077351, -15.2979126, -19.2222309, -15.2938881, -2.5075788, 2.5135922
2: -6.4978967, -3.5649998, -6.5065241, -3.5525074, -1.9956188, 2.0003862
3: -10.7870522, -7.8204222, -10.7924032, -7.8054309, -2.4212923, 2.4116268
4: -13.5415668, -10.6120348, -13.5617371, -10.6064196, -2.2714562, 2.2843328
5: -4.6162295, -2.1803131, -4.6266246, -2.1723967, -1.7677612, 1.7693558
6: -4.4798203, -1.9806833, -4.4906611, -1.9710909, -2.0556579, 2.0611424
7: -12.7794056, -8.8369770, -12.7977867, -8.8146200, -3.0267739, 3.0229068
8: -5.4262915, -3.1817360, -5.4326797, -3.1724625, -1.4726593, 1.4739509
9: -1.8517418, 1.0188844, -1.8735580, 1.0315781, -2.6748400, 2.6839995

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523210, upper bound: 1.0466941
time: 6.13 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523190, upper bound: 1.0471076
time: 5.64 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 7.7806282, 10.1899796, 7.7558002, 10.2273102, -2.0727015, 2.0662684
1: -19.2077351, -15.2979126, -19.2571220, -15.2724323, -2.5291567, 2.5478778
2: -6.4978967, -3.5649998, -6.5187736, -3.5493991, -1.9994745, 2.0146613
3: -10.7870522, -7.8204222, -10.8182745, -7.7942362, -2.4327335, 2.4362369
4: -13.5415668, -10.6120348, -13.5826235, -10.5927801, -2.2840452, 2.3051577
5: -4.6162295, -2.1803131, -4.6398072, -2.1623821, -1.7769489, 1.7824452
6: -4.4798203, -1.9806833, -4.5130687, -1.9215262, -2.0916514, 2.0840440
7: -12.7794056, -8.8369770, -12.8134775, -8.7832298, -3.0572720, 3.0364466
8: -5.4262915, -3.1817360, -5.4488106, -3.1507573, -1.4907334, 1.4906898
9: -1.8517418, 1.0188844, -1.9213347, 1.0464048, -2.6902590, 2.7322717

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523190, upper bound: 1.0466961
time: 7.46 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523200, upper bound: 1.0471091
time: 7.42 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7653236, 10.1997185, 7.7702398, 10.2073517, -2.0729914, 2.0737882
1: -19.2253227, -15.2469673, -19.2222214, -15.2938919, -2.5243855, 2.5540133
2: -6.5393338, -3.5509324, -6.5065222, -3.5525150, -2.0362659, 2.0145779
3: -10.8433599, -7.8034406, -10.7924023, -7.8054385, -2.4588604, 2.4288335
4: -13.5508528, -10.5742722, -13.5617313, -10.6064196, -2.2807398, 2.3129749
5: -4.6317458, -2.1358287, -4.6266170, -2.1724005, -1.7824845, 1.7979088
6: -4.4913344, -1.9598174, -4.4906554, -1.9710923, -2.0880532, 2.0818009
7: -12.8587818, -8.8074350, -12.7977839, -8.8146324, -3.0708199, 3.0598645
8: -5.4382496, -3.1745210, -5.4326768, -3.1724625, -1.4975660, 1.4811950
9: -1.9152765, 1.0279934, -1.8735538, 1.0315721, -2.7393503, 2.6931958

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477549, upper bound: 1.0471075
time: 5.51 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477549, upper bound: 1.0471073
time: 5.41 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7653236, 10.1997185, 7.7558045, 10.2273111, -2.0871234, 2.0845685
1: -19.2253227, -15.2469673, -19.2571144, -15.2724333, -2.5459614, 2.5708671
2: -6.5393338, -3.5509324, -6.5187716, -3.5494084, -2.0413661, 2.0288513
3: -10.8433599, -7.8034406, -10.8182716, -7.7942414, -2.4638762, 2.4534435
4: -13.5508528, -10.5742722, -13.5826216, -10.5927801, -2.2933283, 2.3228443
5: -4.6317458, -2.1358287, -4.6397972, -2.1623869, -1.7916727, 1.8062410
6: -4.4913344, -1.9598174, -4.5130649, -1.9215283, -2.1064761, 2.0984926
7: -12.8587818, -8.8074350, -12.8134747, -8.7832394, -3.0871582, 3.0734034
8: -5.4382496, -3.1745210, -5.4488068, -3.1507607, -1.5054646, 1.4979329
9: -1.9152765, 1.0279934, -1.9213295, 1.0463996, -2.7442703, 2.7414660

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4575
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477549, upper bound: 1.0471092
time: 5.66 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477549, upper bound: 1.0471092
time: 6.29 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 7.7686567, 10.2174139, 7.7697930, 10.2005768, -2.0642309, 2.0748310
1: -19.2248135, -15.2929382, -19.2277756, -15.2819252, -2.5328608, 2.5281005
2: -6.5115609, -3.5520177, -6.5051451, -3.5633140, -2.0069361, 2.0037975
3: -10.7929115, -7.8039832, -10.8020439, -7.8105998, -2.4237051, 2.4317160
4: -13.5695877, -10.6058292, -13.5529547, -10.6018410, -2.3036323, 2.2830014
5: -4.6272163, -2.1694930, -4.6233015, -2.1733124, -1.7768545, 1.7767618
6: -4.4925151, -1.9654703, -4.5003710, -1.9544973, -2.0887623, 2.0826755
7: -12.8078575, -8.8138428, -12.7926998, -8.8191719, -3.0479102, 3.0418754
8: -5.4340730, -3.1679649, -5.4392595, -3.1700392, -1.4871860, 1.4904027
9: -1.8838558, 1.0316219, -1.8780870, 1.0331306, -2.7091007, 2.7014589

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477530, upper bound: 1.0516731
time: 5.43 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477530, upper bound: 1.0516750
time: 6.04 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 7.7686615, 10.2174139, 7.7545571, 10.2103014, -2.0781183, 2.0891953
1: -19.2248058, -15.2929420, -19.2453499, -15.2311296, -2.5705757, 2.5449018
2: -6.5115576, -3.5520296, -6.5466866, -3.5492535, -2.0211029, 2.0429835
3: -10.7929077, -7.8039932, -10.8583317, -7.7936039, -2.4409118, 2.4678380
4: -13.5695820, -10.6058331, -13.5622416, -10.5641403, -2.3249664, 2.2922873
5: -4.6272116, -2.1694980, -4.6388130, -2.1288564, -1.8066366, 1.7915444
6: -4.4925122, -1.9654717, -4.5119185, -1.9335923, -2.1064713, 2.1099877
7: -12.8078556, -8.8138523, -12.8720531, -8.7896156, -3.0851707, 3.0835433
8: -5.4340701, -3.1679678, -5.4511542, -3.1628342, -1.4944234, 1.5112635
9: -1.8838496, 1.0316162, -1.9417543, 1.0422392, -2.7182922, 2.7557030

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477530, upper bound: 1.0520835
time: 5.37 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0477540, upper bound: 1.0520837
time: 7.54 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7708049, 10.2161932, 7.7576723, 10.2280045, -2.0682764, 2.0696287
1: -19.2140694, -15.2937984, -19.2448311, -15.2768831, -2.5234756, 2.5369911
2: -6.5105438, -3.5616088, -6.5188684, -3.5503397, -2.0129681, 2.0103838
3: -10.7906675, -7.8155823, -10.8083506, -7.7941914, -2.4512081, 2.4402933
4: -13.5628271, -10.6069460, -13.5810108, -10.5955544, -2.2906671, 2.2983413
5: -4.6185913, -2.1722479, -4.6342840, -2.1624134, -1.7847943, 1.7895172
6: -4.4865618, -1.9660454, -4.5130773, -1.9392816, -2.0874791, 2.0865583
7: -12.8053646, -8.8308916, -12.8211403, -8.7960443, -3.0492001, 3.0356388
8: -5.4319687, -3.1694789, -5.4470520, -3.1562681, -1.4901185, 1.4914501
9: -1.8788447, 1.0231292, -1.9102092, 1.0459368, -2.7128916, 2.7203379

Time for backsubstitution: 14.64 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.1192193031311035
rel_dist={0: [-1.0573893980579818, 1.057389512159725]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062315, upper bound: 0.9092262
time: 8.63 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094646, upper bound: 0.9094645
time: 7.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.11 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 16.11
Output dim: 0, lower bound: -0.9062315, upper bound: 0.9092262
IS_B2, status: Status.UNKNOWN, split count: 1, time: 16.11
Output dim: 0, lower bound: -0.9094646, upper bound: 0.9094645

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: 7.7583423, 10.2263012, 7.7684388, 10.2174206, -2.0093207, 2.0086422
1: -19.2421169, -15.2778988, -19.2248173, -15.2928391, -2.3981977, 2.3965268
2: -6.5179663, -3.5506077, -6.5115714, -3.5520163, -1.9175253, 1.9118514
3: -10.8063602, -7.7944450, -10.7933292, -7.8039856, -2.3324637, 2.3365149
4: -13.5792751, -10.5961924, -13.5695906, -10.6057529, -2.1838579, 2.1831961
5: -4.6331682, -2.1629679, -4.6272192, -2.1693683, -1.6991425, 1.7008162
6: -4.5127406, -1.9435682, -4.4925184, -1.9654677, -2.0200939, 2.0217953
7: -12.8206997, -8.7985287, -12.8079023, -8.8138399, -2.9324136, 2.9304328
8: -5.4464722, -3.1581059, -5.4340787, -3.1679492, -1.4175711, 1.4148607
9: -1.9062834, 1.0458262, -1.8838596, 1.0316887, -2.6576662, 2.6496606

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022339, upper bound: 0.9089398
time: 7.87 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062210, upper bound: 0.9092183
time: 6.51 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: 7.7540402, 10.2373829, 7.7540426, 10.2373781, -2.0214620, 2.0339231
1: -19.2597084, -15.2714071, -19.2597046, -15.2714100, -2.4372215, 2.4239235
2: -6.5238466, -3.5489068, -6.5238457, -3.5489109, -1.9291372, 1.9289041
3: -10.8192253, -7.7928085, -10.8192225, -7.7928071, -2.3700881, 2.3615279
4: -13.5905037, -10.5921230, -13.5904970, -10.5921268, -2.2073431, 2.1927261
5: -4.6404037, -2.1593943, -4.6404028, -2.1593966, -1.7160425, 1.7138948
6: -4.5149169, -1.9158931, -4.5149150, -1.9159002, -2.0253797, 2.0725417
7: -12.8235588, -8.7824402, -12.8235588, -8.7824440, -2.9439573, 2.9628015
8: -5.4501810, -3.1462440, -5.4501791, -3.1462469, -1.4229410, 1.4434509
9: -1.9316535, 1.0465147, -1.9316473, 1.0465150, -2.6988401, 2.6767368

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4575
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4575

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054688, upper bound: 0.9091779
time: 6.27 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094560, upper bound: 0.9094563
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.10 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 26.10
Output dim: 0, lower bound: -0.9022339, upper bound: 0.9089398
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 26.10
Output dim: 0, lower bound: -0.9062210, upper bound: 0.9092183
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 26.10
Output dim: 0, lower bound: -0.9054688, upper bound: 0.9091779
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 26.10
Output dim: 0, lower bound: -0.9094560, upper bound: 0.9094563

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: 7.7605109, 10.2140274, 7.7782664, 10.1912088, -1.9802651, 1.9862795
1: -19.2389679, -15.2791634, -19.2184887, -15.2969704, -2.3900423, 2.3936753
2: -6.5118265, -3.5512085, -6.4989195, -3.5554147, -1.9078317, 1.8896513
3: -10.8051996, -7.7961988, -10.7896709, -7.8088303, -2.3243380, 2.3288798
4: -13.5696859, -10.5970011, -13.5483303, -10.6108570, -2.1669488, 2.1613379
5: -4.6324415, -2.1666446, -4.6248550, -2.1774464, -1.6887422, 1.6939867
6: -4.5104737, -1.9504182, -4.4857645, -1.9801078, -2.0058079, 2.0059462
7: -12.8084583, -8.7994900, -12.7819271, -8.8199348, -2.9125843, 2.9036007
8: -5.4447837, -3.1635914, -5.4283876, -3.1802011, -1.4035537, 1.3988178
9: -1.8937197, 1.0456924, -1.8567619, 1.0274441, -2.6415310, 2.6222448

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022334, upper bound: 0.9087294
time: 5.72 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022334, upper bound: 0.9089395
time: 5.83 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: 7.7583432, 10.2262993, 7.7684426, 10.2174129, -1.9868279, 2.0086408
1: -19.2421188, -15.2778969, -19.2248154, -15.2928429, -2.3980589, 2.3926086
2: -6.5179667, -3.5506091, -6.5115700, -3.5520153, -1.9175229, 1.9052660
3: -10.8063612, -7.7944446, -10.7933302, -7.8039842, -2.3417988, 2.3343863
4: -13.5792761, -10.5961914, -13.5695877, -10.6057549, -2.1832900, 2.1697464
5: -4.6331682, -2.1629682, -4.6272178, -2.1693721, -1.6984401, 1.7039146
6: -4.5127406, -1.9435695, -4.4925199, -1.9654692, -2.0105572, 2.0211244
7: -12.8206978, -8.7985315, -12.8078947, -8.8138428, -2.9321823, 2.9106045
8: -5.4464731, -3.1581068, -5.4340749, -3.1679506, -1.4077892, 1.4148586
9: -1.9062824, 1.0458264, -1.8838563, 1.0316887, -2.6576653, 2.6454439

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9090078
time: 5.65 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9092179
time: 6.29 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: 7.7561884, 10.2251101, 7.7637668, 10.2111721, -1.9923716, 2.0116658
1: -19.2565613, -15.2726498, -19.2533894, -15.2754498, -2.4291382, 2.4210992
2: -6.5176935, -3.5495083, -6.5111165, -3.5523076, -1.9194317, 1.9066122
3: -10.8180676, -7.7945490, -10.8155174, -7.7976027, -2.3619833, 2.3538394
4: -13.5809031, -10.5929251, -13.5691833, -10.5972061, -2.1903963, 2.1707816
5: -4.6396780, -2.1630440, -4.6380377, -2.1673660, -1.7057834, 1.7070403
6: -4.5126586, -1.9227461, -4.5081902, -1.9305443, -2.0110950, 2.0567112
7: -12.8113241, -8.7834005, -12.7976437, -8.7885475, -2.9241514, 2.9360304
8: -5.4485073, -3.1517329, -5.4444995, -3.1584921, -1.4089358, 1.4274724
9: -1.9190817, 1.0463812, -1.9045305, 1.0422702, -2.6826878, 2.6492405

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9052833, upper bound: 0.9053067
time: 5.50 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054662, upper bound: 0.9091750
time: 7.50 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: 7.7540407, 10.2373829, 7.7540431, 10.2373714, -1.9989696, 2.0339212
1: -19.2597084, -15.2714071, -19.2597008, -15.2714119, -2.4370708, 2.4200573
2: -6.5238466, -3.5489082, -6.5238419, -3.5489097, -1.9291363, 1.9223192
3: -10.8192263, -7.7928061, -10.8192215, -7.7928095, -2.3794379, 2.3594112
4: -13.5905037, -10.5921230, -13.5904942, -10.5921278, -2.2067742, 2.1792397
5: -4.6404047, -2.1593943, -4.6404009, -2.1593976, -1.7153282, 1.7170405
6: -4.5149174, -1.9158934, -4.5149145, -1.9159026, -2.0158801, 2.0718665
7: -12.8235607, -8.7824373, -12.8235540, -8.7824469, -2.9437199, 2.9430227
8: -5.4501810, -3.1462469, -5.4501781, -3.1462479, -1.4131591, 1.4434495
9: -1.9316530, 1.0465145, -1.9316421, 1.0465150, -2.6988363, 2.6725235

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092705, upper bound: 0.9055850
time: 6.26 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094534, upper bound: 0.9094533
time: 6.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.14 seconds
IS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9022334, upper bound: 0.9087294
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9022334, upper bound: 0.9089395
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9090078
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9092179
IS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9052833, upper bound: 0.9053067
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9054662, upper bound: 0.9091750
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9092705, upper bound: 0.9055850
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.14
Output dim: 0, lower bound: -0.9094534, upper bound: 0.9094533

## BFS IS instance: IS_B1_B1_B1

### Backsubstitution after applying IS history:
0: 7.7609844, 10.2137833, 7.7806282, 10.1899796, -1.9748130, 1.9795523
1: -19.2368088, -15.2793560, -19.2077351, -15.2979126, -2.3867788, 2.3822966
2: -6.5116215, -3.5531297, -6.4978967, -3.5649998, -1.8980818, 1.8868706
3: -10.8046761, -7.7985249, -10.7870522, -7.8204222, -2.3124666, 2.3242168
4: -13.5683298, -10.5972376, -13.5415668, -10.6120348, -2.1647820, 2.1543837
5: -4.6307135, -2.1672215, -4.6162295, -2.1803131, -1.6840572, 1.6847429
6: -4.5092702, -1.9505336, -4.4798203, -1.9806833, -1.9998460, 1.9962459
7: -12.8079548, -8.8029060, -12.7794056, -8.8369770, -2.8940911, 2.8955240
8: -5.4443603, -3.1638966, -5.4262915, -3.1817360, -1.3980794, 1.3935025
9: -1.8927164, 1.0439754, -1.8517418, 1.0188844, -2.6321363, 2.6163650

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_B1_B1_B1_A1

### Relational analysis result of IS_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9063975
time: 6.14 seconds

## Relational analysis of IS_B1_B1_B1_A2

### Relational analysis result of IS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9087295
time: 6.32 seconds

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: 7.7605128, 10.2140255, 7.7653236, 10.1997185, -1.9971404, 1.9948649
1: -19.2389641, -15.2791615, -19.2253227, -15.2469673, -2.4230366, 2.3980441
2: -6.5118260, -3.5512195, -6.5393338, -3.5509324, -1.9112000, 1.9312682
3: -10.8051968, -7.7962074, -10.8433599, -7.8034406, -2.3289986, 2.3622012
4: -13.5696812, -10.5969992, -13.5508528, -10.5742722, -2.1906414, 2.1629753
5: -4.6324339, -2.1666498, -4.6317458, -2.1358287, -1.7129469, 1.6991353
6: -4.5104675, -1.9504194, -4.4913344, -1.9598174, -2.0222836, 2.0181484
7: -12.8084545, -8.7995033, -12.8587818, -8.8074350, -2.9278674, 2.9399843
8: -5.4447803, -3.1635952, -5.4382496, -3.1745210, -1.4062033, 1.4137604
9: -1.8937140, 1.0456853, -1.9152765, 1.0279934, -2.6415653, 2.6825843

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_B1_B1_B2_A1

### Relational analysis result of IS_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9066080
time: 5.40 seconds

## Relational analysis of IS_B1_B1_B2_A2

### Relational analysis result of IS_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9089391
time: 11.19 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: 7.7588167, 10.2260561, 7.7708049, 10.2161932, -1.9813857, 2.0019150
1: -19.2399616, -15.2780924, -19.2140694, -15.2937984, -2.3947840, 2.3812213
2: -6.5177598, -3.5525298, -6.5105438, -3.5616088, -1.9077606, 1.9024920
3: -10.8058348, -7.7967715, -10.7906675, -7.8155823, -2.3299127, 2.3296609
4: -13.5779181, -10.5964317, -13.5628271, -10.6069460, -2.1811061, 2.1627913
5: -4.6314411, -2.1635456, -4.6185913, -2.1722479, -1.6938133, 1.6946702
6: -4.5115371, -1.9436848, -4.4865618, -1.9660454, -2.0045953, 2.0113893
7: -12.8201933, -8.8019438, -12.8053646, -8.8308916, -2.9136715, 2.9025259
8: -5.4460487, -3.1584105, -5.4319687, -3.1694789, -1.4023163, 1.4095407
9: -1.9052811, 1.0441110, -1.8788447, 1.0231292, -2.6482630, 2.6395741

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_B1_B2_B1_A1

### Relational analysis result of IS_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9066759
time: 5.50 seconds

## Relational analysis of IS_B1_B2_B1_A2

### Relational analysis result of IS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9090077
time: 6.49 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: 7.7583456, 10.2262983, 7.7554874, 10.2259350, -2.0035815, 2.0172119
1: -19.2421074, -15.2779007, -19.2316551, -15.2428761, -2.4308910, 2.3969622
2: -6.5179653, -3.5506172, -6.5520196, -3.5475361, -1.9208822, 1.9458303
3: -10.8063564, -7.7944551, -10.8470058, -7.7986088, -2.3464494, 2.3673127
4: -13.5792713, -10.5961943, -13.5721264, -10.5691853, -2.2080212, 2.1713910
5: -4.6331606, -2.1629725, -4.6340756, -2.1277943, -1.7226810, 1.7090361
6: -4.5127339, -1.9435698, -4.4981117, -1.9451681, -2.0270233, 2.0352678
7: -12.8206959, -8.7985401, -12.8846989, -8.8013744, -2.9474850, 2.9457326
8: -5.4464664, -3.1581087, -5.4439526, -3.1622705, -1.4104400, 1.4315299
9: -1.9062757, 1.0458207, -1.9424324, 1.0322385, -2.6576900, 2.7052865

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9068861
time: 5.58 seconds

## Relational analysis of IS_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9092178
time: 5.93 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: 7.7507553, 10.2255821, 7.7637787, 10.2111645, -1.9976892, 2.0083408
1: -19.2590294, -15.2659531, -19.2533607, -15.2754517, -2.4280362, 2.4276457
2: -6.5205822, -3.5485191, -6.5111151, -3.5524163, -1.9204097, 1.9053514
3: -10.8210821, -7.7849030, -10.8155060, -7.7977219, -2.3620572, 2.3636856
4: -13.5813370, -10.5878687, -13.5689182, -10.5972071, -2.1871328, 2.1727314
5: -4.6400690, -2.1589489, -4.6379342, -2.1673675, -1.7018051, 1.7048769
6: -4.5307002, -1.9217162, -4.5081825, -1.9305621, -2.0286412, 2.0470963
7: -12.8177853, -8.7821712, -12.7976236, -8.7885551, -2.9206524, 2.9309707
8: -5.4621348, -3.1508422, -5.4444990, -3.1585984, -1.4223070, 1.4209166
9: -1.9217978, 1.0568831, -1.9045196, 1.0422394, -2.6824236, 2.6599607

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054658, upper bound: 0.9089643
time: 6.18 seconds

## Relational analysis of IS_B2_B1_A2_B2

### Relational analysis result of IS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054658, upper bound: 0.9091746
time: 6.17 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: 7.7556782, 10.2316856, 7.7550163, 10.2339821, -1.9933519, 2.0271888
1: -19.2533798, -15.2738924, -19.2559319, -15.2728901, -2.4290743, 2.4134398
2: -6.5225019, -3.5524790, -6.5230427, -3.5519915, -1.9258876, 1.9176402
3: -10.8098383, -7.7941279, -10.8136368, -7.7935915, -2.3687792, 2.3520589
4: -13.5849323, -10.5930052, -13.5871792, -10.5926504, -2.2003965, 2.1740136
5: -4.6357174, -2.1609697, -4.6376119, -2.1603351, -1.7088404, 1.7106905
6: -4.5128355, -1.9327741, -4.5136776, -1.9259491, -2.0031447, 2.0533872
7: -12.8226585, -8.7883577, -12.8230181, -8.7859688, -2.9403911, 2.9352102
8: -5.4487209, -3.1583295, -5.4493113, -3.1534386, -1.4042757, 1.4303389
9: -1.9189310, 1.0461750, -1.9240742, 1.0463150, -2.6857729, 2.6645055

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092700, upper bound: 0.9053743
time: 5.70 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092700, upper bound: 0.9055844
time: 5.56 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7486148, 10.2378521, 7.7540531, 10.2373638, -2.0042276, 2.0305901
1: -19.2621803, -15.2647095, -19.2596741, -15.2714119, -2.4359884, 2.4266014
2: -6.5266995, -3.5479155, -6.5238414, -3.5490210, -1.9300723, 1.9210482
3: -10.8222685, -7.7831631, -10.8192139, -7.7929287, -2.3794470, 2.3692570
4: -13.5909405, -10.5870676, -13.5902357, -10.5921268, -2.2035108, 2.1811671
5: -4.6407948, -2.1552992, -4.6402960, -2.1594000, -1.7113466, 1.7148089
6: -4.5329580, -1.9148659, -4.5149040, -1.9159192, -2.0334177, 2.0622549
7: -12.8300104, -8.7812052, -12.8235350, -8.7824535, -2.9402657, 2.9379654
8: -5.4638071, -3.1453614, -5.4501781, -3.1463552, -1.4265299, 1.4369037
9: -1.9343653, 1.0570180, -1.9316339, 1.0464828, -2.6985683, 2.6832423

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6123
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6123

## Relational analysis of IS_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094529, upper bound: 0.9092425
time: 5.13 seconds

## Relational analysis of IS_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094529, upper bound: 0.9094524
time: 5.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.36 seconds
IS_B1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9063975
IS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9087295
IS_B1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9066080
IS_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9089391
IS_B1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9066759
IS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9090077
IS_B1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9068861
IS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9062206, upper bound: 0.9092178
IS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9054658, upper bound: 0.9089643
IS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9054658, upper bound: 0.9091746
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9092700, upper bound: 0.9053743
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9092700, upper bound: 0.9055844
IS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9094529, upper bound: 0.9092425
IS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 0, lower bound: -0.9094529, upper bound: 0.9094524

## BFS IS instance: IS_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 7.7566624, 10.2248611, 7.7806282, 10.1899796, -1.9793873, 1.9834886
1: -19.2543964, -15.2728519, -19.2077351, -15.2979126, -2.4042368, 2.3886919
2: -6.5174837, -3.5514324, -6.4978967, -3.5649998, -1.9055829, 1.8896344
3: -10.8175411, -7.7968750, -10.7870522, -7.8204222, -2.3287506, 2.3232365
4: -13.5795422, -10.5931702, -13.5415668, -10.6120348, -2.1756635, 2.1576805
5: -4.6379485, -2.1636248, -4.6162295, -2.1803131, -1.6918073, 1.6869044
6: -4.5114546, -1.9228702, -4.4798203, -1.9806833, -2.0025611, 2.0072932
7: -12.8108158, -8.7868195, -12.7794056, -8.8369770, -2.8940320, 2.9144120
8: -5.4480829, -3.1520381, -5.4262915, -3.1817360, -1.4023190, 1.4002424
9: -1.9180708, 1.0446639, -1.8517418, 1.0188844, -2.6578588, 2.6172342

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_B1_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9050212
time: 5.53 seconds

## Relational analysis of IS_B1_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9087295
time: 5.46 seconds

## BFS IS instance: IS_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 7.7561932, 10.2251034, 7.7653236, 10.1997185, -1.9971576, 1.9988718
1: -19.2565460, -15.2726536, -19.2253227, -15.2469673, -2.4270558, 2.4044437
2: -6.5176883, -3.5495191, -6.5393338, -3.5509324, -1.9187016, 1.9319241
3: -10.8180599, -7.7945604, -10.8433599, -7.8034406, -2.3452806, 2.3548760
4: -13.5808926, -10.5929279, -13.5508528, -10.5742722, -2.1931539, 2.1662769
5: -4.6396675, -2.1630516, -4.6317458, -2.1358287, -1.7162318, 1.7013042
6: -4.5126519, -1.9227539, -4.4913344, -1.9598174, -2.0177510, 2.0215478
7: -12.8113165, -8.7834158, -12.8587818, -8.8074350, -2.9278121, 2.9449477
8: -5.4485054, -3.1517367, -5.4382496, -3.1745210, -1.4104476, 1.4151174
9: -1.9190693, 1.0463731, -1.9152765, 1.0279934, -2.6672945, 2.6721601

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_B1_B1_B2_A2_A1

### Relational analysis result of IS_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9052311
time: 8.83 seconds

## Relational analysis of IS_B1_B1_B2_A2_A2

### Relational analysis result of IS_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9022335, upper bound: 0.9089391
time: 10.62 seconds

## BFS IS instance: IS_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 7.7545128, 10.2371330, 7.7708049, 10.2161932, -1.9859400, 2.0105181
1: -19.2575493, -15.2716084, -19.2140694, -15.2937984, -2.4122410, 2.3876381
2: -6.5236373, -3.5508337, -6.5105438, -3.5616088, -1.9152737, 1.9052610
3: -10.8186951, -7.7951336, -10.7906675, -7.8155823, -2.3462176, 2.3286777
4: -13.5891438, -10.5923710, -13.5628271, -10.6069460, -2.1919899, 2.1661077
5: -4.6386747, -2.1599762, -4.6185913, -2.1722479, -1.7015586, 1.6968558
6: -4.5137105, -1.9160161, -4.4865618, -1.9660454, -2.0073304, 2.0243955
7: -12.8230534, -8.7858582, -12.8053646, -8.8308916, -2.9136066, 2.9213657
8: -5.4497557, -3.1465526, -5.4319687, -3.1694789, -1.4065468, 1.4180248
9: -1.9306426, 1.0447991, -1.8788447, 1.0231292, -2.6740093, 2.6404433

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_B1_B2_B1_A2_A1

### Relational analysis result of IS_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9060098, upper bound: 0.9090076
time: 5.73 seconds

## Relational analysis of IS_B1_B2_B1_A2_A2

### Relational analysis result of IS_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9060098, upper bound: 0.9090075
time: 6.19 seconds

## BFS IS instance: IS_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 7.7540464, 10.2373753, 7.7554874, 10.2259350, -2.0035973, 2.0258906
1: -19.2596951, -15.2714100, -19.2316551, -15.2428761, -2.4349113, 2.4033828
2: -6.5238423, -3.5489202, -6.5520196, -3.5475361, -1.9283962, 1.9464927
3: -10.8192167, -7.7928176, -10.8470058, -7.7986088, -2.3627553, 2.3599958
4: -13.5904932, -10.5921268, -13.5721264, -10.5691853, -2.2105446, 2.1747127
5: -4.6403942, -2.1594005, -4.6340756, -2.1277943, -1.7259626, 1.7112164
6: -4.5149083, -1.9159002, -4.4981117, -1.9451681, -2.0221372, 2.0386710
7: -12.8235579, -8.7824583, -12.8846989, -8.8013744, -2.9474239, 2.9506502
8: -5.4501743, -3.1462493, -5.4439526, -3.1622705, -1.4146752, 1.4328890
9: -1.9316401, 1.0465093, -1.9424324, 1.0322385, -2.6834393, 2.6948643

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 5736

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 900

## Relational analysis of IS_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_B1_B2_B2_A2_A1

### Relational analysis result of IS_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054435, upper bound: 0.9091996
time: 5.76 seconds

## Relational analysis of IS_B1_B2_B2_A2_A2

### Relational analysis result of IS_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9062015, upper bound: 0.9091993
time: 5.65 seconds

## BFS IS instance: IS_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 7.7512307, 10.2253380, 7.7661285, 10.2099400, -1.9922395, 2.0016232
1: -19.2568779, -15.2661495, -19.2426224, -15.2764378, -2.4247417, 2.4162760
2: -6.5203757, -3.5504422, -6.5100780, -3.5619938, -1.9106512, 1.9025574
3: -10.8205633, -7.7872286, -10.8129196, -7.8093152, -2.3501797, 2.3590298
4: -13.5799780, -10.5881138, -13.5621529, -10.5984182, -2.1849208, 2.1657696
5: -4.6383419, -2.1595283, -4.6293249, -2.1702580, -1.6970854, 1.6956422
6: -4.5294952, -1.9218322, -4.5022078, -1.9311416, -2.0226903, 2.0373569
7: -12.8172741, -8.7855816, -12.7950878, -8.8055563, -2.9021502, 2.9228525
8: -5.4617114, -3.1511488, -5.4424028, -3.1601291, -1.4168286, 1.4155600
9: -1.9207978, 1.0551658, -1.8994846, 1.0336771, -2.6730213, 2.6540461

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9054658, upper bound: 0.9052562
time: 6.89 seconds

## Relational analysis of IS_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054657, upper bound: 0.9089642
time: 6.23 seconds

## BFS IS instance: IS_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 7.7507620, 10.2255821, 7.7509365, 10.2196608, -2.0093212, 2.0168061
1: -19.2590256, -15.2659512, -19.2601910, -15.2256603, -2.4584994, 2.4320087
2: -6.5205798, -3.5485291, -6.5516891, -3.5479283, -1.9237676, 1.9449759
3: -10.8210783, -7.7849126, -10.8692398, -7.7923117, -2.3667274, 2.3844750
4: -13.5813313, -10.5878687, -13.5714417, -10.5607128, -2.2098579, 2.1743708
5: -4.6400609, -2.1589522, -4.6448102, -2.1257989, -1.7280264, 1.7100325
6: -4.5306950, -1.9217172, -4.5137415, -1.9102125, -2.0379584, 2.0527496
7: -12.8177795, -8.7821827, -12.8744392, -8.7760410, -2.9363742, 2.9642563
8: -5.4621315, -3.1508465, -5.4542351, -3.1529164, -1.4241328, 1.4327793
9: -1.9217916, 1.0568740, -1.9632759, 1.0427830, -2.6824484, 2.7092285

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4575

## Relational analysis of IS_B2_B1_A2_B2_A1

### Relational analysis result of IS_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9054658, upper bound: 0.9054664
time: 5.34 seconds

## Relational analysis of IS_B2_B1_A2_B2_A2

### Relational analysis result of IS_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054657, upper bound: 0.9091745
time: 6.02 seconds

## BFS IS instance: IS_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 7.7561493, 10.2314415, 7.7573652, 10.2327614, -1.9879079, 2.0204725
1: -19.2512226, -15.2740898, -19.2451935, -15.2738752, -2.4257679, 2.4020605
2: -6.5222950, -3.5544071, -6.5220089, -3.5615885, -1.9161205, 1.9148540
3: -10.8093128, -7.7964554, -10.8110018, -7.8051925, -2.3568902, 2.3473549
4: -13.5835781, -10.5932503, -13.5804167, -10.5938711, -2.1981740, 2.1670585
5: -4.6339908, -2.1615496, -4.6289983, -2.1632338, -1.7042079, 1.7014623
6: -4.5116334, -1.9328880, -4.5076900, -1.9265283, -1.9971938, 2.0436163
7: -12.8221550, -8.7917728, -12.8204927, -8.8029823, -2.9218454, 2.9270902
8: -5.4482951, -3.1586342, -5.4472008, -3.1549664, -1.3987999, 1.4249840
9: -1.9179282, 1.0444584, -1.9190526, 1.0377526, -2.6763687, 2.6585994

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_B2_B2_A1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9090592, upper bound: 0.9053738
time: 5.61 seconds

## Relational analysis of IS_B2_B2_A1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9090592, upper bound: 0.9053737
time: 5.30 seconds

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 7.7556834, 10.2316837, 7.7420893, 10.2424908, -2.0087347, 2.0356698
1: -19.2533722, -15.2738953, -19.2627678, -15.2231188, -2.4594240, 2.4177856
2: -6.5224996, -3.5524900, -6.5636520, -3.5475013, -1.9292660, 1.9546528
3: -10.8098364, -7.7941394, -10.8673306, -7.7881880, -2.3734417, 2.3799720
4: -13.5849285, -10.5930052, -13.5897198, -10.5561657, -2.2232442, 2.1756587
5: -4.6357088, -2.1609731, -4.6444588, -2.1188016, -1.7326486, 1.7157564
6: -4.5128307, -1.9327743, -4.5192633, -1.9055989, -2.0196996, 2.0609312
7: -12.8226547, -8.7883711, -12.8998260, -8.7734852, -2.9559212, 2.9640856
8: -5.4487171, -3.1583314, -5.4590387, -3.1477623, -1.4069252, 1.4439503
9: -1.9189277, 1.0461674, -1.9828591, 1.0468569, -2.6857958, 2.7212377

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 5732
type: B, layer: 1, pos: 5732
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_B2_B2_A1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084930, upper bound: 0.9055662
time: 6.46 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092509, upper bound: 0.9055659
time: 6.71 seconds

## BFS IS instance: IS_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 7.7490864, 10.2376080, 7.7564044, 10.2361450, -1.9987822, 2.0238738
1: -19.2600212, -15.2649136, -19.2489376, -15.2724075, -2.4326787, 2.4152222
2: -6.5264945, -3.5498364, -6.5228033, -3.5585980, -1.9203067, 1.9182608
3: -10.8217459, -7.7854872, -10.8165817, -7.8045263, -2.3675580, 2.3645530
4: -13.5895834, -10.5873146, -13.5834713, -10.5933514, -2.2012825, 2.1742029
5: -4.6390676, -2.1558795, -4.6316833, -2.1623020, -1.7067251, 1.7055795
6: -4.5317540, -1.9149823, -4.5089154, -1.9164985, -2.0274668, 2.0524874
7: -12.8295069, -8.7846174, -12.8210077, -8.7994614, -2.9217472, 2.9298410
8: -5.4633789, -3.1456647, -5.4480705, -3.1478782, -1.4210534, 1.4315450
9: -1.9333649, 1.0553010, -1.9266100, 1.0379207, -2.6891623, 2.6773362

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6123
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 4575
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5732
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5732
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6123

## Relational analysis of IS_B2_B2_A2_B1_A1

### Relational analysis result of IS_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092422, upper bound: 0.9092425
time: 5.56 seconds

## Relational analysis of IS_B2_B2_A2_B1_A2

### Relational analysis result of IS_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092422, upper bound: 0.9092420
time: 4.90 seconds

## BFS IS instance: IS_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 7.7486210, 10.2378540, 7.7411280, 10.2458754, -2.0157485, 2.0390425
1: -19.2621689, -15.2647142, -19.2665100, -15.2216425, -2.4663115, 2.4309497
2: -6.5266967, -3.5479281, -6.5644488, -3.5445318, -1.9334211, 1.9596224
3: -10.8222637, -7.7831731, -10.8729172, -7.7875209, -2.3841105, 2.3896503
4: -13.5909357, -10.5870695, -13.5927753, -10.5556459, -2.2272720, 2.1828113
5: -4.6407857, -2.1553035, -4.6471376, -2.1178694, -1.7376945, 1.7199359
6: -4.5329523, -1.9148669, -4.5204868, -1.8955606, -2.0423532, 2.0698690
7: -12.8300095, -8.7812176, -12.9003448, -8.7699738, -2.9560213, 2.9699936
8: -5.4637995, -3.1453638, -5.4598851, -3.1406741, -1.4283943, 1.4505162
9: -1.9343581, 1.0570109, -1.9904490, 1.0470240, -2.6985865, 2.7319560

Time for backsubstitution: 14.52 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.0339293479919434
rel_dist={0: [-0.9094671207244254, 0.9094665870331546]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2411.29 seconds
