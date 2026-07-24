## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.294357096
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9367981, -7.0425968, -8.9367981, -7.0425968, -1.8942013, 1.8942013)
1: (2.4124451, 3.9131384, 2.4124451, 3.9131384, -1.4570764, 1.4570764)
2: (-6.5181541, -4.9777999, -6.5181541, -4.9777999, -1.4703059, 1.4703059)
3: (-11.4270353, -9.5017948, -11.4270353, -9.5017948, -1.8738117, 1.8738117)
4: (-4.3657799, -2.8065395, -4.3657799, -2.8065395, -1.5592403, 1.5592403)
5: (-12.3431225, -10.5792007, -12.3431225, -10.5792007, -1.6700945, 1.6700947)
6: (-10.0652647, -8.0891485, -10.0652647, -8.0891485, -1.9091916, 1.9091921)
7: (-4.2142544, -2.6923499, -4.2142544, -2.6923499, -1.3958604, 1.3958603)
8: (-3.2913580, -1.8388863, -3.2913580, -1.8388863, -1.3433269, 1.3433267)
9: (-12.0051117, -10.4397650, -12.0051117, -10.4397650, -1.5272553, 1.5272552)

## BASE Result
execution time: IAR + LP analysis = 15.07 + 31.80 = 46.87 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3553.13 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.1101056337356567

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9366202354431152
rel_dist={1: [-0.5181458851188188, 0.518142788570696]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8209632635116577
rel_dist={1: [-0.29885889706970215, 0.2988555407171436]}

## Binary Search Result
Binary search time: 147.49 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 3405.64 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=None

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5112041, upper bound: 0.5181389
time: 3.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181377, upper bound: 0.5181391
time: 4.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.32
Output dim: 1, lower bound: -0.5112041, upper bound: 0.5181389
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.32
Output dim: 1, lower bound: -0.5181377, upper bound: 0.5181391

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9359207, -7.0585661, -8.9362888, -7.0519896, -1.2259293, 1.2190423
1: 2.4402661, 3.9130864, 2.4288263, 3.9131081, -0.9081157, 0.9198204
2: -6.5177808, -4.9914556, -6.5179396, -4.9858332, -0.9290211, 0.9231154
3: -11.4268007, -9.5475321, -11.4268990, -9.5287027, -1.0976036, 1.0785265
4: -4.3217554, -2.8069339, -4.3398786, -2.8067675, -1.0558739, 1.0747354
5: -12.3428230, -10.5806141, -12.3429470, -10.5800352, -0.9370219, 0.9366271
6: -10.0495138, -8.0893326, -10.0559959, -8.0892534, -1.1068573, 1.1135962
7: -4.2139525, -2.7113905, -4.2140799, -2.7035518, -0.8132999, 0.8052244
8: -3.2912493, -1.8574257, -3.2912946, -1.8497934, -0.7099417, 0.7022642
9: -12.0047808, -10.4753914, -12.0049200, -10.4607239, -0.8652499, 0.8503439

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109394, upper bound: 0.5109388
time: 4.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109394, upper bound: 0.5181404
time: 4.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.9613400, -7.0414629, -8.9367981, -7.0426021, -1.2509327, 1.2325573
1: 2.4103026, 3.9472642, 2.4124551, 3.9131386, -0.9203613, 0.9475802
2: -6.5370874, -4.9757576, -6.5181546, -4.9778090, -0.9442365, 0.9331371
3: -11.4848194, -9.5002756, -11.4270363, -9.5018044, -1.1411412, 1.0933900
4: -4.3692021, -2.7483325, -4.3657713, -2.8065424, -1.0811605, 1.1126879
5: -12.3488350, -10.5776911, -12.3431263, -10.5792007, -0.9475729, 0.9414518
6: -10.0702209, -8.0683899, -10.0652590, -8.0891495, -1.1202731, 1.1388216
7: -4.2392397, -2.6909769, -4.2142544, -2.6923594, -0.8353376, 0.8170779
8: -3.3150744, -1.8385506, -3.2913561, -1.8388929, -0.7330920, 0.7082822
9: -12.0504303, -10.4382887, -12.0051107, -10.4397821, -0.8973837, 0.8651665

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5109361
time: 5.11 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5181371
time: 5.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.94 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.94
Output dim: 1, lower bound: -0.5109394, upper bound: 0.5109388
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.94
Output dim: 1, lower bound: -0.5109394, upper bound: 0.5181404
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.94
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5109361
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.94
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5181371

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.9359207, -7.0585661, -8.9359207, -7.0585661, -1.2172277, 1.2172277
1: 2.4402661, 3.9130864, 2.4402661, 3.9130864, -0.9080935, 0.9080935
2: -6.5177808, -4.9914556, -6.5177808, -4.9914556, -0.9221590, 0.9221591
3: -11.4268007, -9.5475321, -11.4268007, -9.5475321, -1.0784292, 1.0784292
4: -4.3217554, -2.8069339, -4.3217554, -2.8069339, -1.0544207, 1.0544205
5: -12.3428230, -10.5806141, -12.3428230, -10.5806141, -0.9356447, 0.9356449
6: -10.0495138, -8.0893326, -10.0495138, -8.0893326, -1.1068079, 1.1068076
7: -4.2139525, -2.7113905, -4.2139525, -2.7113905, -0.8043624, 0.8043623
8: -3.2912493, -1.8574257, -3.2912493, -1.8574257, -0.7019458, 0.7019458
9: -12.0047808, -10.4753914, -12.0047808, -10.4753914, -0.8496823, 0.8496822

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5109308
time: 3.95 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111942, upper bound: 0.5109291
time: 4.11 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9359207, -7.0585661, -8.9613400, -7.0414629, -1.2330728, 1.2297857
1: 2.4402661, 3.9130864, 2.4103026, 3.9472642, -0.9190943, 0.9297121
2: -6.5177808, -4.9914556, -6.5370874, -4.9757576, -0.9332360, 0.9275608
3: -11.4268007, -9.5475321, -11.4848194, -9.5002756, -1.1079090, 1.0945388
4: -4.3217554, -2.8069339, -4.3692021, -2.7483325, -1.0633483, 1.0815332
5: -12.3428230, -10.5806141, -12.3488350, -10.5776911, -0.9390674, 0.9420445
6: -10.0495138, -8.0893326, -10.0702209, -8.0683899, -1.1223469, 1.1283095
7: -4.2139525, -2.7113905, -4.2392397, -2.6909769, -0.8192794, 0.8136196
8: -3.2912493, -1.8574257, -3.3150744, -1.8385506, -0.7184814, 0.7136592
9: -12.0047808, -10.4753914, -12.0504303, -10.4382887, -0.8708000, 0.8595364

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5181318
time: 3.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111942, upper bound: 0.5181301
time: 4.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.9613400, -7.0414629, -8.9359207, -7.0585661, -1.2297854, 1.2330728
1: 2.4103026, 3.9472642, 2.4402661, 3.9130864, -0.9297122, 0.9190944
2: -6.5370874, -4.9757576, -6.5177808, -4.9914556, -0.9275609, 0.9332360
3: -11.4848194, -9.5002756, -11.4268007, -9.5475321, -1.0945389, 1.1079091
4: -4.3692021, -2.7483325, -4.3217554, -2.8069339, -1.0815331, 1.0633482
5: -12.3488350, -10.5776911, -12.3428230, -10.5806141, -0.9420444, 0.9390671
6: -10.0702209, -8.0683899, -10.0495138, -8.0893326, -1.1283095, 1.1223466
7: -4.2392397, -2.6909769, -4.2139525, -2.7113905, -0.8136197, 0.8192797
8: -3.3150744, -1.8385506, -3.2912493, -1.8574257, -0.7136592, 0.7184814
9: -12.0504303, -10.4382887, -12.0047808, -10.4753914, -0.8595366, 0.8708000

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118720, upper bound: 0.5109313
time: 4.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181296, upper bound: 0.5109303
time: 4.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.9613400, -7.0414629, -8.9613400, -7.0414629, -1.2341502, 1.2341504
1: 2.4103026, 3.9472642, 2.4103026, 3.9472642, -0.9206989, 0.9206989
2: -6.5370874, -4.9757576, -6.5370874, -4.9757576, -0.9333849, 0.9333849
3: -11.4848194, -9.5002756, -11.4848194, -9.5002756, -1.0947468, 1.0947469
4: -4.3692021, -2.7483325, -4.3692021, -2.7483325, -1.0815067, 1.0815067
5: -12.3488350, -10.5776911, -12.3488350, -10.5776911, -0.9496874, 0.9496875
6: -10.0702209, -8.0683899, -10.0702209, -8.0683899, -1.1225338, 1.1225339
7: -4.2392397, -2.6909769, -4.2392397, -2.6909769, -0.8172745, 0.8172743
8: -3.3150744, -1.8385506, -3.3150744, -1.8385506, -0.7088989, 0.7088989
9: -12.0504303, -10.4382887, -12.0504303, -10.4382887, -0.8658714, 0.8658713

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5109315
time: 4.08 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181307, upper bound: 0.5109297
time: 6.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5109308
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5111942, upper bound: 0.5109291
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5181318
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5111942, upper bound: 0.5181301
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5118720, upper bound: 0.5109313
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5181296, upper bound: 0.5109303
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5109315
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.88
Output dim: 1, lower bound: -0.5181307, upper bound: 0.5109297

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9348497, -7.0589561, -1.2132285, 1.2137637
1: 2.4457912, 3.9116077, 2.4420075, 3.9125435, -0.9001633, 0.9049135
2: -6.5152645, -5.0031300, -6.5170116, -4.9951348, -0.9164813, 0.9098448
3: -11.4202719, -9.5516996, -11.4247446, -9.5488081, -1.0696025, 1.0727303
4: -4.3153510, -2.8075817, -4.3197384, -2.8071301, -1.0454247, 1.0504839
5: -12.3369751, -10.5831194, -12.3409834, -10.5813866, -0.9282745, 0.9309692
6: -10.0451155, -8.0904264, -10.0481071, -8.0896759, -1.1012006, 1.1030351
7: -4.2133045, -2.7129230, -4.2137489, -2.7118526, -0.8025823, 0.8027933
8: -3.2897959, -1.8628788, -3.2908077, -1.8591471, -0.6982039, 0.6945596
9: -12.0044041, -10.4814310, -12.0046654, -10.4773054, -0.8463564, 0.8414969

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5049335
time: 3.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5111955
time: 3.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9791784, -7.0411239, -8.9359150, -7.0585661, -1.2585807, 1.2292719
1: 2.4296799, 3.9674935, 2.4402795, 3.9130833, -0.9280361, 0.9374263
2: -6.6075277, -4.9887009, -6.5177765, -4.9914780, -0.9509203, 0.9223883
3: -11.4273491, -9.4510326, -11.4267817, -9.5475397, -1.0809700, 1.1372893
4: -4.3280263, -2.7707949, -4.3217416, -2.8069355, -1.0665878, 1.0679539
5: -12.3439198, -10.5245686, -12.3428011, -10.5806179, -0.9351028, 0.9585032
6: -10.0780430, -8.0592976, -10.0495062, -8.0893345, -1.1496539, 1.1252453
7: -4.2237558, -2.6855869, -4.2139511, -2.7113938, -0.8167311, 0.8295829
8: -3.3455305, -1.8536944, -3.2912474, -1.8574352, -0.7287186, 0.7103605
9: -12.0337296, -10.4644232, -12.0047808, -10.4754047, -0.8609847, 0.8705745

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5049320
time: 3.75 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5111937
time: 3.66 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9602642, -7.0418530, -1.2290049, 1.2262897
1: 2.4457912, 3.9116077, 2.4120431, 3.9467220, -0.9111391, 0.9261451
2: -6.5152645, -5.0031300, -6.5363197, -4.9794364, -0.9268291, 0.9151683
3: -11.4202719, -9.5516996, -11.4827642, -9.5015516, -1.0990691, 1.0884938
4: -4.3153510, -2.8075817, -4.3671870, -2.7485275, -1.0542984, 1.0772507
5: -12.3369751, -10.5831194, -12.3469954, -10.5784674, -0.9316982, 0.9373652
6: -10.0451155, -8.0904264, -10.0688152, -8.0687304, -1.1165967, 1.1245605
7: -4.2133045, -2.7129230, -4.2390366, -2.6914389, -0.8175185, 0.8120255
8: -3.2897959, -1.8628788, -3.3146319, -1.8402672, -0.7144439, 0.7062532
9: -12.0044041, -10.4814310, -12.0503111, -10.4402027, -0.8671877, 0.8513817

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5118712
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5181318
time: 3.98 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9791784, -7.0411239, -8.9613323, -7.0414639, -1.2648840, 1.2322905
1: 2.4296799, 3.9674935, 2.4103165, 3.9472609, -0.9290307, 0.9483795
2: -6.6075277, -4.9887009, -6.5370841, -4.9757805, -0.9575427, 0.9279042
3: -11.4273491, -9.4510326, -11.4848003, -9.5002832, -1.1061254, 1.1386579
4: -4.3280263, -2.7707949, -4.3691893, -2.7483327, -1.0681596, 1.0877099
5: -12.3439198, -10.5245686, -12.3488178, -10.5776968, -0.9385250, 0.9620014
6: -10.0780430, -8.0592976, -10.0702114, -8.0683918, -1.1539483, 1.1369240
7: -4.2237558, -2.6855869, -4.2392406, -2.6909800, -0.8244971, 0.8306221
8: -3.3455305, -1.8536944, -3.3150721, -1.8385572, -0.7343825, 0.7154583
9: -12.0337296, -10.4644232, -12.0504274, -10.4383011, -0.8734763, 0.8718026

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5118691
time: 4.06 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5181304
time: 4.08 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9348497, -7.0589561, -1.2257175, 1.2295766
1: 2.4158239, 3.9457893, 2.4420075, 3.9125435, -0.9217387, 0.9155396
2: -6.5345716, -4.9874301, -6.5170116, -4.9951348, -0.9211533, 0.9208450
3: -11.4782887, -9.5044403, -11.4247446, -9.5488081, -1.0856981, 1.1018617
4: -4.3628030, -2.7489786, -4.3197384, -2.8071301, -1.0724938, 1.0590631
5: -12.3429804, -10.5802021, -12.3409834, -10.5813866, -0.9346656, 0.9343972
6: -10.0658302, -8.0694809, -10.0481071, -8.0896759, -1.1227760, 1.1184673
7: -4.2385912, -2.6925077, -4.2137489, -2.7118526, -0.8118575, 0.8176980
8: -3.3136191, -1.8440018, -3.2908077, -1.8591471, -0.7096221, 0.7110751
9: -12.0500526, -10.4443245, -12.0046654, -10.4773054, -0.8559229, 0.8626491

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118694, upper bound: 0.5049338
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118720, upper bound: 0.5111972
time: 4.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.0046387, -7.0240340, -8.9359150, -7.0585661, -1.2615824, 1.2355592
1: 2.3997457, 4.0016723, 2.4402795, 3.9130833, -0.9396085, 0.9377636
2: -6.6268344, -4.9730048, -6.5177765, -4.9914780, -0.9518673, 0.9335778
3: -11.4853697, -9.4038057, -11.4267817, -9.5475397, -1.0927548, 1.1520193
4: -4.3754568, -2.7121925, -4.3217416, -2.8069355, -1.0863181, 1.0695251
5: -12.3499451, -10.5216694, -12.3428011, -10.5806179, -0.9415063, 0.9620359
6: -10.0986595, -8.0383520, -10.0495062, -8.0893345, -1.1648383, 1.1260376
7: -4.2490463, -2.6651800, -4.2139511, -2.7113938, -0.8188359, 0.8363136
8: -3.3693542, -1.8348169, -3.2912474, -1.8574352, -0.7295558, 0.7202802
9: -12.0793762, -10.4273252, -12.0047808, -10.4754047, -0.8622125, 0.8831235

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181325, upper bound: 0.5049316
time: 4.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181325, upper bound: 0.5111934
time: 4.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9602642, -7.0418530, -1.2301474, 1.2306859
1: 2.4158239, 3.9457893, 2.4120431, 3.9467220, -0.9127601, 0.9175113
2: -6.5345716, -4.9874301, -6.5363197, -4.9794364, -0.9277053, 0.9210662
3: -11.4782887, -9.5044403, -11.4827642, -9.5015516, -1.0859203, 1.0890424
4: -4.3628030, -2.7489786, -4.3671870, -2.7485275, -1.0725102, 1.0775698
5: -12.3429804, -10.5802021, -12.3469954, -10.5784674, -0.9423087, 0.9447786
6: -10.0658302, -8.0694809, -10.0688152, -8.0687304, -1.1170082, 1.1187874
7: -4.2385912, -2.6925077, -4.2390366, -2.6914389, -0.8154935, 0.8157151
8: -3.3136191, -1.8440018, -3.3146319, -1.8402672, -0.7051587, 0.7015132
9: -12.0500526, -10.4443245, -12.0503111, -10.4402027, -0.8625453, 0.8576854

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5046694
time: 4.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5109325
time: 4.42 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0046387, -7.0240340, -8.9613323, -7.0414639, -1.2722979, 1.2430458
1: 2.3997457, 4.0016723, 2.4103165, 3.9472609, -0.9401896, 0.9489579
2: -6.6268344, -4.9730048, -6.5370841, -4.9757805, -0.9610090, 0.9336215
3: -11.4853697, -9.4038057, -11.4848003, -9.5002832, -1.0972884, 1.1536379
4: -4.3754568, -2.7121925, -4.3691893, -2.7483327, -1.0919418, 1.0933228
5: -12.3499451, -10.5216694, -12.3488178, -10.5776968, -0.9491680, 0.9681132
6: -10.0986595, -8.0383520, -10.0702114, -8.0683918, -1.1656315, 1.1380367
7: -4.2490463, -2.6651800, -4.2392406, -2.6909800, -0.8275143, 0.8393352
8: -3.3693542, -1.8348169, -3.3150721, -1.8385572, -0.7360139, 0.7173131
9: -12.0793762, -10.4273252, -12.0504274, -10.4383011, -0.8764899, 0.8861399

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181335, upper bound: 0.5046675
time: 4.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181312, upper bound: 0.5109288
time: 4.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5049335
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5111955
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5049320
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5111937
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5118712
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5049355, upper bound: 0.5181318
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5118691
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5111972, upper bound: 0.5181304
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5118694, upper bound: 0.5049338
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5118720, upper bound: 0.5111972
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5181325, upper bound: 0.5049316
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5181325, upper bound: 0.5111934
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5046694
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5109325
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5181335, upper bound: 0.5046675
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.43
Output dim: 1, lower bound: -0.5181312, upper bound: 0.5109288

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9323978, -7.0598145, -1.2115338, 1.2115338
1: 2.4457912, 3.9116077, 2.4457912, 3.9116077, -0.8996335, 0.8996334
2: -6.5152645, -5.0031300, -6.5152645, -5.0031300, -0.9084525, 0.9084524
3: -11.4202719, -9.5516996, -11.4202719, -9.5516996, -1.0673785, 1.0673785
4: -4.3153510, -2.8075817, -4.3153510, -2.8075817, -1.0445879, 1.0445876
5: -12.3369751, -10.5831194, -12.3369751, -10.5831194, -0.9264660, 0.9264660
6: -10.0451155, -8.0904264, -10.0451155, -8.0904264, -1.0996509, 1.0996510
7: -4.2133045, -2.7129230, -4.2133045, -2.7129230, -0.8018039, 0.8018038
8: -3.2897959, -1.8628788, -3.2897959, -1.8628788, -0.6934633, 0.6934633
9: -12.0044041, -10.4814310, -12.0044041, -10.4814310, -0.8409138, 0.8409140

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5049363
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5049337
time: 4.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9705391, -7.0411749, -1.2252569, 1.2479694
1: 2.4457912, 3.9116077, 2.4298329, 3.9611762, -0.9247091, 0.9170374
2: -6.5152645, -5.0031300, -6.5999379, -4.9887409, -0.9231868, 0.9343060
3: -11.4202719, -9.5516996, -11.4273462, -9.4622269, -1.1192901, 1.0750047
4: -4.3153510, -2.8075817, -4.3278570, -2.7732477, -1.0577033, 1.0581963
5: -12.3369751, -10.5831194, -12.3439178, -10.5294371, -0.9492397, 0.9340808
6: -10.0451155, -8.0904264, -10.0710955, -8.0593176, -1.1201725, 1.1298139
7: -4.2133045, -2.7129230, -4.2236958, -2.6900687, -0.8235016, 0.8125308
8: -3.2897959, -1.8628788, -3.3394365, -1.8537369, -0.7028098, 0.7176513
9: -12.0044041, -10.4814310, -12.0336838, -10.4667158, -0.8565860, 0.8530635

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5111964
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5111942
time: 4.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9705391, -7.0411749, -8.9323978, -7.0598145, -1.2479692, 1.2252572
1: 2.4298329, 3.9611762, 2.4457912, 3.9116077, -0.9170371, 0.9247092
2: -6.5999379, -4.9887409, -6.5152645, -5.0031300, -0.9343061, 0.9231868
3: -11.4273462, -9.4622269, -11.4202719, -9.5516996, -1.0750048, 1.1192902
4: -4.3278570, -2.7732477, -4.3153510, -2.8075817, -1.0581963, 1.0577033
5: -12.3439178, -10.5294371, -12.3369751, -10.5831194, -0.9340811, 0.9492397
6: -10.0710955, -8.0593176, -10.0451155, -8.0904264, -1.1298141, 1.1201725
7: -4.2236958, -2.6900687, -4.2133045, -2.7129230, -0.8125306, 0.8235017
8: -3.3394365, -1.8537369, -3.2897959, -1.8628788, -0.7176514, 0.7028098
9: -12.0336838, -10.4667158, -12.0044041, -10.4814310, -0.8530636, 0.8565860

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106868, upper bound: 0.5049320
time: 3.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111930, upper bound: 0.5049322
time: 4.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9874249, -7.0410767, -8.9874249, -7.0410767, -1.2690115, 1.2690113
1: 2.4295378, 3.9733124, 2.4295378, 3.9733124, -0.9521561, 0.9521561
2: -6.6145153, -4.9886618, -6.6145153, -4.9886618, -0.9558890, 0.9558889
3: -11.4273510, -9.4406414, -11.4273510, -9.4406414, -1.1449680, 1.1449680
4: -4.3281822, -2.7685292, -4.3281822, -2.7685292, -1.0745100, 1.0745101
5: -12.3439198, -10.5196209, -12.3439198, -10.5196209, -0.9607971, 0.9607972
6: -10.0847330, -8.0592766, -10.0847330, -8.0592766, -1.1651082, 1.1651084
7: -4.2238207, -2.6814618, -4.2238207, -2.6814618, -0.8394349, 0.8394349
8: -3.3511605, -1.8536549, -3.3511605, -1.8536549, -0.7346531, 0.7346531
9: -12.0337706, -10.4622965, -12.0337706, -10.4622965, -0.8765221, 0.8765223

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106876, upper bound: 0.5049320
time: 4.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111939, upper bound: 0.5049318
time: 4.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9578161, -7.0427113, -1.2273421, 1.2240548
1: 2.4457912, 3.9116077, 2.4158239, 3.9457893, -0.9105737, 0.9211733
2: -6.5152645, -5.0031300, -6.5345716, -4.9874301, -0.9193211, 0.9136442
3: -11.4202719, -9.5516996, -11.4782887, -9.5044403, -1.0968542, 1.0834868
4: -4.3153510, -2.8075817, -4.3628030, -2.7489786, -1.0534841, 1.0716791
5: -12.3369751, -10.5831194, -12.3429804, -10.5802021, -0.9298940, 0.9328572
6: -10.0451155, -8.0904264, -10.0658302, -8.0694809, -1.1150575, 1.1212262
7: -4.2133045, -2.7129230, -4.2385912, -2.6925077, -0.8167527, 0.8110800
8: -3.2897959, -1.8628788, -3.3136191, -1.8440018, -0.7099934, 0.7051716
9: -12.0044041, -10.4814310, -12.0500526, -10.4443245, -0.8620696, 0.8508019

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5118727
time: 4.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5118714
time: 4.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9960136, -7.0240803, -1.2315474, 1.2509804
1: 2.4457912, 3.9116077, 2.3998928, 3.9953547, -0.9250467, 0.9370824
2: -6.5152645, -5.0031300, -6.6192446, -4.9730434, -0.9302616, 0.9352529
3: -11.4202719, -9.5516996, -11.4853649, -9.4149990, -1.1340225, 1.0888581
4: -4.3153510, -2.8075817, -4.3752928, -2.7146440, -1.0592744, 1.0835648
5: -12.3369751, -10.5831194, -12.3499470, -10.5265408, -0.9527771, 0.9401213
6: -10.0451155, -8.0904264, -10.0917206, -8.0383711, -1.1209650, 1.1513708
7: -4.2133045, -2.7129230, -4.2489834, -2.6696601, -0.8302257, 0.8172642
8: -3.2897959, -1.8628788, -3.3632617, -1.8348598, -0.7176689, 0.7184900
9: -12.0044041, -10.4814310, -12.0793295, -10.4296160, -0.8768150, 0.8542920

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5181316
time: 4.28 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5181294
time: 4.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9705391, -7.0411749, -8.9578161, -7.0427113, -1.2542734, 1.2282739
1: 2.4298329, 3.9611762, 2.4158239, 3.9457893, -0.9265053, 0.9356446
2: -6.5999379, -4.9887409, -6.5345716, -4.9874301, -0.9409299, 0.9245870
3: -11.4273462, -9.4622269, -11.4782887, -9.5044403, -1.1022263, 1.1206589
4: -4.3278570, -2.7732477, -4.3628030, -2.7489786, -1.0654000, 1.0774703
5: -12.3439178, -10.5294371, -12.3429804, -10.5802021, -0.9375091, 0.9527379
6: -10.0710955, -8.0593176, -10.0658302, -8.0694809, -1.1424360, 1.1318779
7: -4.2236958, -2.6900687, -4.2385912, -2.6925077, -0.8229368, 0.8245401
8: -3.3394365, -1.8537369, -3.3136191, -1.8440018, -0.7233152, 0.7128474
9: -12.0336838, -10.4667158, -12.0500526, -10.4443245, -0.8655593, 0.8654909

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106842, upper bound: 0.5118684
time: 4.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111930, upper bound: 0.5118679
time: 4.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9874249, -7.0410767, -9.0128641, -7.0239873, -1.2752962, 1.2720065
1: 2.4295378, 3.9733124, 2.3996077, 4.0074911, -0.9524904, 0.9630675
2: -6.6145153, -4.9886618, -6.6338215, -4.9729686, -0.9625089, 0.9568359
3: -11.4273510, -9.4406414, -11.4853725, -9.3934155, -1.1596961, 1.1463367
4: -4.3281822, -2.7685292, -4.3756084, -2.7099276, -1.0760809, 1.0942369
5: -12.3439198, -10.5196209, -12.3499451, -10.5167255, -0.9643319, 0.9643049
6: -10.0847330, -8.0592766, -10.1053467, -8.0383301, -1.1659007, 1.1767795
7: -4.2238207, -2.6814618, -4.2491102, -2.6610548, -0.8461701, 0.8404737
8: -3.3511605, -1.8536549, -3.3749824, -1.8347774, -0.7403166, 0.7354884
9: -12.0337706, -10.4622965, -12.0794172, -10.4251995, -0.8890388, 0.8777508

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106851, upper bound: 0.5118691
time: 4.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111939, upper bound: 0.5118684
time: 4.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9323978, -7.0598145, -1.2240553, 1.2273421
1: 2.4158239, 3.9457893, 2.4457912, 3.9116077, -0.9211733, 0.9105737
2: -6.5345716, -4.9874301, -6.5152645, -5.0031300, -0.9136441, 0.9193208
3: -11.4782887, -9.5044403, -11.4202719, -9.5516996, -1.0834868, 1.0968542
4: -4.3628030, -2.7489786, -4.3153510, -2.8075817, -1.0716791, 1.0534841
5: -12.3429804, -10.5802021, -12.3369751, -10.5831194, -0.9328570, 0.9298940
6: -10.0658302, -8.0694809, -10.0451155, -8.0904264, -1.1212263, 1.1150575
7: -4.2385912, -2.6925077, -4.2133045, -2.7129230, -0.8110800, 0.8167527
8: -3.3136191, -1.8440018, -3.2897959, -1.8628788, -0.7051716, 0.7099934
9: -12.0500526, -10.4443245, -12.0044041, -10.4814310, -0.8508019, 0.8620698

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113282, upper bound: 0.5049326
time: 4.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118685, upper bound: 0.5049353
time: 4.09 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9705391, -7.0411749, -1.2282739, 1.2542734
1: 2.4158239, 3.9457893, 2.4298329, 3.9611762, -0.9356446, 0.9265051
2: -6.5345716, -4.9874301, -6.5999379, -4.9887409, -0.9245872, 0.9409299
3: -11.4782887, -9.5044403, -11.4273462, -9.4622269, -1.1206591, 1.1022261
4: -4.3628030, -2.7489786, -4.3278570, -2.7732477, -1.0774703, 1.0653999
5: -12.3429804, -10.5802021, -12.3439178, -10.5294371, -0.9527380, 0.9375089
6: -10.0658302, -8.0694809, -10.0710955, -8.0593176, -1.1318779, 1.1424358
7: -4.2385912, -2.6925077, -4.2236958, -2.6900687, -0.8245401, 0.8229367
8: -3.3136191, -1.8440018, -3.3394365, -1.8537369, -0.7128474, 0.7233151
9: -12.0500526, -10.4443245, -12.0336838, -10.4667158, -0.8654909, 0.8655593

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113282, upper bound: 0.5111929
time: 3.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118685, upper bound: 0.5111962
time: 4.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9960136, -7.0240803, -8.9323978, -7.0598145, -1.2509804, 1.2315476
1: 2.3998928, 3.9953547, 2.4457912, 3.9116077, -0.9370825, 0.9250464
2: -6.6192446, -4.9730434, -6.5152645, -5.0031300, -0.9352529, 0.9302615
3: -11.4853649, -9.4149990, -11.4202719, -9.5516996, -1.0888581, 1.1340226
4: -4.3752928, -2.7146440, -4.3153510, -2.8075817, -1.0835648, 1.0592744
5: -12.3499470, -10.5265408, -12.3369751, -10.5831194, -0.9401213, 0.9527771
6: -10.0917206, -8.0383711, -10.0451155, -8.0904264, -1.1513705, 1.1209648
7: -4.2489834, -2.6696601, -4.2133045, -2.7129230, -0.8172643, 0.8302257
8: -3.3632617, -1.8348598, -3.2897959, -1.8628788, -0.7184900, 0.7176689
9: -12.0793295, -10.4296160, -12.0044041, -10.4814310, -0.8542922, 0.8768150

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176175, upper bound: 0.5049310
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181282, upper bound: 0.5049346
time: 4.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0128641, -7.0239873, -8.9874249, -7.0410767, -1.2720065, 1.2752964
1: 2.3996077, 4.0074911, 2.4295378, 3.9733124, -0.9630675, 0.9524906
2: -6.6338215, -4.9729686, -6.6145153, -4.9886618, -0.9568357, 0.9625088
3: -11.4853725, -9.3934155, -11.4273510, -9.4406414, -1.1463368, 1.1596960
4: -4.3756084, -2.7099276, -4.3281822, -2.7685292, -1.0942367, 1.0760809
5: -12.3499451, -10.5167255, -12.3439198, -10.5196209, -0.9643049, 0.9643316
6: -10.1053467, -8.0383301, -10.0847330, -8.0592766, -1.1767797, 1.1659009
7: -4.2491102, -2.6610548, -4.2238207, -2.6814618, -0.8404737, 0.8461702
8: -3.3749824, -1.8347774, -3.3511605, -1.8536549, -0.7354884, 0.7403166
9: -12.0794172, -10.4251995, -12.0337706, -10.4622965, -0.8777509, 0.8890389

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176184, upper bound: 0.5049311
time: 4.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181291, upper bound: 0.5049346
time: 4.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9578161, -7.0427113, -1.2284534, 1.2284536
1: 2.4158239, 3.9457893, 2.4158239, 3.9457893, -0.9122297, 0.9122298
2: -6.5345716, -4.9874301, -6.5345716, -4.9874301, -0.9196737, 0.9196736
3: -11.4782887, -9.5044403, -11.4782887, -9.5044403, -1.0836906, 1.0836905
4: -4.3628030, -2.7489786, -4.3628030, -2.7489786, -1.0716732, 1.0716730
5: -12.3429804, -10.5802021, -12.3429804, -10.5802021, -0.9405261, 0.9405261
6: -10.0658302, -8.0694809, -10.0658302, -8.0694809, -1.1154585, 1.1154584
7: -4.2385912, -2.6925077, -4.2385912, -2.6925077, -0.8147254, 0.8147255
8: -3.3136191, -1.8440018, -3.3136191, -1.8440018, -0.7004176, 0.7004176
9: -12.0500526, -10.4443245, -12.0500526, -10.4443245, -0.8571026, 0.8571026

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113265, upper bound: 0.5046686
time: 4.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118721, upper bound: 0.5046722
time: 4.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9960136, -7.0240803, -1.2390304, 1.2616899
1: 2.4158239, 3.9457893, 2.3998928, 3.9953547, -0.9362233, 0.9296329
2: -6.5345716, -4.9874301, -6.6192446, -4.9730434, -0.9337282, 0.9443945
3: -11.4782887, -9.5044403, -11.4853649, -9.4149990, -1.1356423, 1.0913174
4: -4.3628030, -2.7489786, -4.3752928, -2.7146440, -1.0830810, 1.0852962
5: -12.3429804, -10.5802021, -12.3499470, -10.5265408, -0.9588537, 0.9462198
6: -10.0658302, -8.0694809, -10.0917206, -8.0383711, -1.1329913, 1.1458125
7: -4.2385912, -2.6925077, -4.2489834, -2.6696601, -0.8332462, 0.8254520
8: -3.3136191, -1.8440018, -3.3632617, -1.8348598, -0.7097645, 0.7249478
9: -12.0500526, -10.4443245, -12.0793295, -10.4296160, -0.8728485, 0.8685726

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113265, upper bound: 0.5109282
time: 3.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118721, upper bound: 0.5109287
time: 3.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9960136, -7.0240803, -8.9578161, -7.0427113, -1.2616897, 1.2390306
1: 2.3998928, 3.9953547, 2.4158239, 3.9457893, -0.9296328, 0.9362233
2: -6.6192446, -4.9730434, -6.5345716, -4.9874301, -0.9443946, 0.9337282
3: -11.4853649, -9.4149990, -11.4782887, -9.5044403, -1.0913177, 1.1356423
4: -4.3752928, -2.7146440, -4.3628030, -2.7489786, -1.0852962, 1.0830811
5: -12.3499470, -10.5265408, -12.3429804, -10.5802021, -0.9462198, 0.9588537
6: -10.0917206, -8.0383711, -10.0658302, -8.0694809, -1.1458125, 1.1329913
7: -4.2489834, -2.6696601, -4.2385912, -2.6925077, -0.8254523, 0.8332462
8: -3.3632617, -1.8348598, -3.3136191, -1.8440018, -0.7249477, 0.7097645
9: -12.0793295, -10.4296160, -12.0500526, -10.4443245, -0.8685727, 0.8728486

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176183, upper bound: 0.5046674
time: 4.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181290, upper bound: 0.5046674
time: 4.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0128641, -7.0239873, -9.0128641, -7.0239873, -1.2827148, 1.2827148
1: 2.3996077, 4.0074911, 2.3996077, 4.0074911, -0.9636489, 0.9636490
2: -6.6338215, -4.9729686, -6.6338215, -4.9729686, -0.9659767, 0.9659767
3: -11.4853725, -9.3934155, -11.4853725, -9.3934155, -1.1613145, 1.1613142
4: -4.3756084, -2.7099276, -4.3756084, -2.7099276, -1.0998596, 1.0998596
5: -12.3499451, -10.5167255, -12.3499451, -10.5167255, -0.9704201, 0.9704201
6: -10.1053467, -8.0383301, -10.1053467, -8.0383301, -1.1779180, 1.1779180
7: -4.2491102, -2.6610548, -4.2491102, -2.6610548, -0.8491901, 0.8491901
8: -3.3749824, -1.8347774, -3.3749824, -1.8347774, -0.7419468, 0.7419468
9: -12.0794172, -10.4251995, -12.0794172, -10.4251995, -0.8920562, 0.8920563

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176195, upper bound: 0.5046675
time: 4.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181299, upper bound: 0.5046674
time: 4.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.42 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5049363
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5049337
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5111964
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5111942
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5106868, upper bound: 0.5049320
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5111930, upper bound: 0.5049322
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5106876, upper bound: 0.5049320
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5111939, upper bound: 0.5049318
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5118727
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5118714
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5181316
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5181294
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5106842, upper bound: 0.5118684
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5111930, upper bound: 0.5118679
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5106851, upper bound: 0.5118691
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5111939, upper bound: 0.5118684
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5113282, upper bound: 0.5049326
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5118685, upper bound: 0.5049353
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5113282, upper bound: 0.5111929
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5118685, upper bound: 0.5111962
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5176175, upper bound: 0.5049310
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5181282, upper bound: 0.5049346
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5176184, upper bound: 0.5049311
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5181291, upper bound: 0.5049346
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5113265, upper bound: 0.5046686
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5118721, upper bound: 0.5046722
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5113265, upper bound: 0.5109282
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5118721, upper bound: 0.5109287
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5176183, upper bound: 0.5046674
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5181290, upper bound: 0.5046674
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5176195, upper bound: 0.5046675
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.42
Output dim: 1, lower bound: -0.5181299, upper bound: 0.5046674

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9240828, -7.0637317, -8.9281616, -7.0620742, -1.1914649, 1.1945047
1: 2.4498925, 3.9088790, 2.4480565, 3.9102457, -0.8908216, 0.8931928
2: -6.5107188, -5.0191026, -6.5129738, -5.0115905, -0.8943231, 0.8884116
3: -11.4016037, -9.5619602, -11.4106264, -9.5568771, -1.0443072, 1.0495182
4: -4.3047762, -2.8126082, -4.3098493, -2.8100939, -1.0298903, 1.0329099
5: -12.3255329, -10.5903902, -12.3309412, -10.5867538, -0.9130484, 0.9173043
6: -10.0291052, -8.0937738, -10.0367756, -8.0922422, -1.0778103, 1.0857475
7: -4.2038727, -2.7137659, -4.2084050, -2.7133756, -0.7909832, 0.7954701
8: -3.2858357, -1.8793821, -3.2878013, -1.8716135, -0.6803964, 0.6740489
9: -12.0012283, -10.4942122, -12.0026779, -10.4880352, -0.8277662, 0.8230243

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043981, upper bound: 0.5043979
time: 4.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043981, upper bound: 0.5049366
time: 4.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9425097, -7.0461788, -8.9323921, -7.0598192, -1.2147827, 1.2320712
1: 2.4354179, 3.9369788, 2.4457948, 3.9116061, -0.9179258, 0.9239058
2: -6.5565405, -4.9976711, -6.5152612, -5.0031390, -0.9243951, 0.9063654
3: -11.4216385, -9.4997292, -11.4202614, -9.5517073, -1.0607486, 1.0990037
4: -4.3235703, -2.7945652, -4.3153448, -2.8075848, -1.0518293, 1.0536371
5: -12.3381214, -10.5507507, -12.3369713, -10.5831251, -0.9267601, 0.9521086
6: -10.0585337, -8.0443392, -10.0450974, -8.0904293, -1.1057053, 1.1334109
7: -4.2175345, -2.6903501, -4.2132998, -2.7129238, -0.8011483, 0.8161511
8: -3.3244696, -1.8582959, -3.2897902, -1.8628931, -0.7088971, 0.6887289
9: -12.0434895, -10.4782515, -12.0044012, -10.4814472, -0.8551803, 0.8405964

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9366202354431152
rel_dist={1: [-0.5181458851188188, 0.518142788570696]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965503, upper bound: 0.2988561
time: 4.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988571, upper bound: 0.2988548
time: 4.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 1, lower bound: -0.2965503, upper bound: 0.2988561
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 1, lower bound: -0.2988571, upper bound: 0.2988548

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9359207, -7.0585661, -8.9360609, -7.0560656, -1.0532794, 1.0506597
1: 2.4402661, 3.9130864, 2.4359288, 3.9130940, -0.7924449, 0.7968972
2: -6.5177808, -4.9914556, -6.5178409, -4.9893174, -0.8071792, 0.8049321
3: -11.4268007, -9.5475321, -11.4268379, -9.5403700, -0.9193718, 0.9121163
4: -4.3217554, -2.8069339, -4.3286467, -2.8068700, -0.9500396, 0.9572153
5: -12.3428230, -10.5806141, -12.3428688, -10.5803947, -0.7742355, 0.7740859
6: -10.0495138, -8.0893326, -10.0519791, -8.0893040, -0.9322050, 0.9347693
7: -4.2139525, -2.7113905, -4.2140031, -2.7084093, -0.6816065, 0.6785344
8: -3.2912493, -1.8574257, -3.2912669, -1.8545232, -0.5669467, 0.5640265
9: -12.0047808, -10.4753914, -12.0048351, -10.4698114, -0.7137942, 0.7081239

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2967341
time: 6.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2988550
time: 6.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.9613400, -7.0414629, -8.9367971, -7.0426044, -1.0815654, 1.0614507
1: 2.4103026, 3.9472642, 2.4124596, 3.9131379, -0.7995714, 0.8266258
2: -6.5370874, -4.9757576, -6.5181537, -4.9778113, -0.8256129, 0.8125253
3: -11.4848194, -9.5002756, -11.4270353, -9.5018072, -0.9727674, 0.9184914
4: -4.3692021, -2.7483325, -4.3657684, -2.8065412, -0.9673390, 1.0053623
5: -12.3488350, -10.5776911, -12.3431244, -10.5791988, -0.7844853, 0.7795191
6: -10.0702209, -8.0683899, -10.0652561, -8.0891495, -0.9429080, 0.9617014
7: -4.2392397, -2.6909769, -4.2142539, -2.6923616, -0.7072642, 0.6869798
8: -3.3150744, -1.8385506, -3.2913570, -1.8388963, -0.5934094, 0.5666686
9: -12.0504303, -10.4382887, -12.0051117, -10.4397888, -0.7524595, 0.7164389

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988551, upper bound: 0.2967331
time: 4.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988551, upper bound: 0.2988517
time: 4.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.26
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2967341
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.26
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2988550
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.26
Output dim: 1, lower bound: -0.2988551, upper bound: 0.2967331
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.26
Output dim: 1, lower bound: -0.2988551, upper bound: 0.2988517

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.9333134, -7.0595007, -8.9325371, -7.0573158, -1.0484128, 1.0455780
1: 2.4444103, 3.9119303, 2.4414549, 3.9116163, -0.7859150, 0.7886349
2: -6.5159121, -5.0002294, -6.5153265, -5.0009918, -0.7939937, 0.7941601
3: -11.4218979, -9.5506229, -11.4203081, -9.5445366, -0.9102738, 0.9018699
4: -4.3169527, -2.8074145, -4.3222423, -2.8075171, -0.9423640, 0.9476776
5: -12.3384361, -10.5824785, -12.3370247, -10.5828991, -0.7666955, 0.7655779
6: -10.0461960, -8.0901537, -10.0475807, -8.0903978, -0.9262946, 0.9281759
7: -4.2134676, -2.7125211, -4.2133532, -2.7099414, -0.6794102, 0.6762664
8: -3.2901754, -1.8615208, -3.2898121, -1.8599749, -0.5588742, 0.5572795
9: -12.0044994, -10.4799328, -12.0044575, -10.4758511, -0.7052379, 0.7013518

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2967349
time: 5.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2967364
time: 4.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9359131, -7.0585666, -8.9680204, -7.0386915, -1.0613098, 1.0790963
1: 2.4402864, 3.9130826, 2.4255433, 3.9592307, -0.8099208, 0.8074226
2: -6.5177746, -4.9914856, -6.5976553, -4.9866157, -0.8039427, 0.8261529
3: -11.4267712, -9.5475435, -11.4273815, -9.4585238, -0.9599400, 0.9115189
4: -4.3217344, -2.8069353, -4.3346963, -2.7739413, -0.9588914, 0.9606444
5: -12.3427925, -10.5806208, -12.3439665, -10.5307178, -0.7927163, 0.7716857
6: -10.0495014, -8.0893383, -10.0714111, -8.0592957, -0.9481020, 0.9623753
7: -4.2139521, -2.7113943, -4.2237239, -2.6884735, -0.6974826, 0.6896304
8: -3.2912445, -1.8574386, -3.3375711, -1.8508463, -0.5726420, 0.5837618
9: -12.0047817, -10.4754105, -12.0337210, -10.4618444, -0.7221317, 0.7161797

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2988537
time: 7.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2988552
time: 3.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.9587307, -7.0423994, -8.9332743, -7.0438557, -1.0766621, 1.0563734
1: 2.4144444, 3.9461117, 2.4179850, 3.9116592, -0.7930325, 0.8183160
2: -6.5352211, -4.9845281, -6.5156393, -4.9894834, -0.8122656, 0.8017492
3: -11.4799166, -9.5033665, -11.4205074, -9.5059738, -0.9633889, 0.9082382
4: -4.3644042, -2.7488110, -4.3593655, -2.8071887, -0.9596627, 0.9957778
5: -12.3444395, -10.5795612, -12.3372784, -10.5817080, -0.7767583, 0.7710146
6: -10.0669098, -8.0692053, -10.0608501, -8.0902424, -0.9370615, 0.9549596
7: -4.2387533, -2.6921062, -4.2136040, -2.6938944, -0.7050576, 0.6847111
8: -3.3139982, -1.8426437, -3.2899027, -1.8443480, -0.5853268, 0.5599222
9: -12.0501461, -10.4428263, -12.0047321, -10.4458284, -0.7439344, 0.7096658

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2967328
time: 4.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2967364
time: 6.71 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.9613295, -7.0414667, -8.9687948, -7.0252361, -1.0835004, 1.0869691
1: 2.4103208, 3.9472599, 2.4020784, 3.9592752, -0.8160161, 0.8335674
2: -6.5370831, -4.9757886, -6.5979662, -4.9751120, -0.8224763, 0.8327138
3: -11.4847898, -9.5002880, -11.4275789, -9.4199629, -1.0035806, 0.9178933
4: -4.3691826, -2.7483335, -4.3718128, -2.7736108, -0.9746253, 1.0069170
5: -12.3488054, -10.5777025, -12.3442192, -10.5295296, -0.7993885, 0.7771175
6: -10.0702066, -8.0683908, -10.0846424, -8.0591412, -0.9574533, 0.9818144
7: -4.2392378, -2.6909802, -4.2239761, -2.6724298, -0.7181969, 0.6949677
8: -3.3150706, -1.8385620, -3.3376656, -1.8352184, -0.5924828, 0.5867879
9: -12.0504274, -10.4383059, -12.0339994, -10.4318247, -0.7576790, 0.7238262

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2988516
time: 4.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2988552
time: 4.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2967349
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2967364
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2988537
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2988552
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2967328
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2967364
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2988516
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.39
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2988552

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9325371, -7.0573158, -1.0475860, 1.0449665
1: 2.4457912, 3.9116077, 2.4414549, 3.9116163, -0.7839848, 0.7884372
2: -6.5152645, -5.0031300, -6.5153265, -5.0009918, -0.7934728, 0.7912254
3: -11.4202719, -9.5516996, -11.4203081, -9.5445366, -0.9083209, 0.9010653
4: -4.3153510, -2.8075817, -4.3222423, -2.8075171, -0.9402065, 0.9473817
5: -12.3369751, -10.5831194, -12.3370247, -10.5828991, -0.7650564, 0.7649059
6: -10.0451155, -8.0904264, -10.0475807, -8.0903978, -0.9250479, 0.9276083
7: -4.2133045, -2.7129230, -4.2133532, -2.7099414, -0.6790483, 0.6759758
8: -3.2897959, -1.8628788, -3.2898121, -1.8599749, -0.5584641, 0.5555439
9: -12.0044041, -10.4814310, -12.0044575, -10.4758511, -0.7050261, 0.6993554

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943357
time: 5.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967349
time: 5.80 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9575291, -7.0413122, -8.9325371, -7.0573158, -1.0686240, 1.0562456
1: 2.4301202, 3.9516675, 2.4414549, 3.9116163, -0.8010994, 0.7988821
2: -6.5885110, -4.9888945, -6.5153265, -5.0009918, -0.8099470, 0.8004941
3: -11.4273415, -9.4792500, -11.4203081, -9.5445366, -0.9159420, 0.9358966
4: -4.3275695, -2.7769396, -4.3222423, -2.8075171, -0.9533906, 0.9525685
5: -12.3439207, -10.5368347, -12.3370247, -10.5828991, -0.7719983, 0.7828276
6: -10.0616398, -8.0593510, -10.0475807, -8.0903978, -0.9410005, 0.9446394
7: -4.2235985, -2.6969047, -4.2133532, -2.7099414, -0.6896996, 0.6889151
8: -3.3302660, -1.8539295, -3.2898121, -1.8599749, -0.5739213, 0.5647724
9: -12.0335999, -10.4701977, -12.0044575, -10.4758511, -0.7115955, 0.7107896

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943372
time: 5.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967345
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9576778, -7.0388126, -1.0577312, 1.0671427
1: 2.4457912, 3.9116077, 2.4257841, 3.9516764, -0.7962066, 0.8055534
2: -6.5152645, -5.0031300, -6.5885720, -4.9867549, -0.8018057, 0.8086357
3: -11.4202719, -9.5516996, -11.4273796, -9.4720917, -0.9401960, 0.9086864
4: -4.3153510, -2.8075817, -4.3344612, -2.7768741, -0.9483124, 0.9586229
5: -12.3369751, -10.5831194, -12.3439665, -10.5366192, -0.7829762, 0.7718511
6: -10.0451155, -8.0904264, -10.0640974, -8.0593204, -0.9430270, 0.9435703
7: -4.2133045, -2.7129230, -4.2236471, -2.6939237, -0.6907072, 0.6866270
8: -3.2897959, -1.8628788, -3.3302841, -1.8510261, -0.5676928, 0.5722018
9: -12.0044041, -10.4814310, -12.0336533, -10.4646196, -0.7164593, 0.7082428

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2964544
time: 4.00 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2988543
time: 3.92 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9874249, -7.0410767, -8.9875708, -7.0385771, -1.1011400, 1.0996594
1: 2.4295378, 3.9733124, 2.4252024, 3.9733205, -0.8284075, 0.8310854
2: -6.6145153, -4.9886618, -6.6145778, -4.9865241, -0.8354592, 0.8341473
3: -11.4273510, -9.4406414, -11.4273901, -9.4334803, -0.9777901, 0.9734906
4: -4.3281822, -2.7685292, -4.3350739, -2.7684646, -0.9646355, 0.9688964
5: -12.3439198, -10.5196209, -12.3439665, -10.5194025, -0.7966237, 0.7964773
6: -10.0847330, -8.0592766, -10.0871887, -8.0592480, -0.9874334, 0.9890556
7: -4.2238207, -2.6814618, -4.2238693, -2.6784797, -0.7131226, 0.7113312
8: -3.3511605, -1.8536549, -3.3511767, -1.8507524, -0.5940942, 0.5923747
9: -12.0337706, -10.4622965, -12.0338230, -10.4567184, -0.7325227, 0.7291687

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2943372
time: 3.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2967364
time: 4.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9332743, -7.0438557, -1.0758333, 1.0557621
1: 2.4158239, 3.9457893, 2.4179850, 3.9116592, -0.7911023, 0.8181055
2: -6.5345716, -4.9874301, -6.5156393, -4.9894834, -0.8116949, 0.7988136
3: -11.4782887, -9.5044403, -11.4205074, -9.5059738, -0.9617119, 0.9074345
4: -4.3628030, -2.7489786, -4.3593655, -2.8071887, -0.9575052, 0.9954958
5: -12.3429804, -10.5802021, -12.3372784, -10.5817080, -0.7753175, 0.7703440
6: -10.0658302, -8.0694809, -10.0608501, -8.0902424, -0.9358320, 0.9543960
7: -4.2385912, -2.6925077, -4.2136040, -2.6938944, -0.7047246, 0.6844310
8: -3.3136191, -1.8440018, -3.2899027, -1.8443480, -0.5849220, 0.5581866
9: -12.0500526, -10.4443245, -12.0047321, -10.4458284, -0.7437241, 0.7076699

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2943378, upper bound: 0.2943372
time: 4.98 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967369, upper bound: 0.2943336
time: 4.28 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9830284, -7.0242138, -8.9332743, -7.0438557, -1.0908461, 1.0641036
1: 2.4001737, 3.9858458, 2.4179850, 3.9116592, -0.8082112, 0.8250208
2: -6.6078167, -4.9731927, -6.5156393, -4.9894834, -0.8260028, 0.8070546
3: -11.4853611, -9.4320202, -11.4205074, -9.5059738, -0.9643831, 0.9425311
4: -4.3750114, -2.7183342, -4.3593655, -2.8071887, -0.9700873, 0.9988256
5: -12.3499432, -10.5339375, -12.3372784, -10.5817080, -0.7787002, 0.7885072
6: -10.0822086, -8.0384064, -10.0608501, -8.0902424, -0.9519525, 0.9602592
7: -4.2488880, -2.6764944, -4.2136040, -2.6938944, -0.7108340, 0.6942770
8: -3.3540945, -1.8350530, -3.2899027, -1.8443480, -0.5923920, 0.5674152
9: -12.0792465, -10.4330978, -12.0047321, -10.4458284, -0.7471440, 0.7191552

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2943378, upper bound: 0.2943367
time: 4.71 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967369, upper bound: 0.2943372
time: 8.03 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9584522, -7.0253544, -1.0799198, 1.0749981
1: 2.4158239, 3.9457893, 2.4023170, 3.9517212, -0.8022842, 0.8319881
2: -6.5345716, -4.9874301, -6.5888834, -4.9752502, -0.8178635, 0.8151964
3: -11.4782887, -9.5044403, -11.4275742, -9.4335318, -0.9838326, 0.9150562
4: -4.3628030, -2.7489786, -4.3715782, -2.7765439, -0.9640549, 1.0048928
5: -12.3429804, -10.5802021, -12.3442183, -10.5354347, -0.7896527, 0.7774913
6: -10.0658302, -8.0694809, -10.0773554, -8.0591679, -0.9524045, 0.9696190
7: -4.2385912, -2.6925077, -4.2238994, -2.6778805, -0.7114210, 0.6936698
8: -3.3136191, -1.8440018, -3.3303776, -1.8353982, -0.5904797, 0.5752275
9: -12.0500526, -10.4443245, -12.0339317, -10.4346008, -0.7525187, 0.7158929

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943372, upper bound: 0.2964554
time: 4.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967364, upper bound: 0.2964527
time: 4.32 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0128641, -7.0239873, -8.9883480, -7.0251226, -1.1233082, 1.1075244
1: 2.3996077, 4.0074911, 2.4017382, 3.9733641, -0.8344669, 0.8572330
2: -6.6338215, -4.9729686, -6.6148911, -4.9750214, -0.8515177, 0.8407075
3: -11.4853725, -9.3934155, -11.4275885, -9.3949127, -1.0214307, 0.9801170
4: -4.3756084, -2.7099276, -4.3721886, -2.7681358, -0.9803495, 1.0151746
5: -12.3499451, -10.5167255, -12.3442183, -10.5182161, -0.8033028, 0.8021593
6: -10.1053467, -8.0383301, -10.1004362, -8.0590935, -0.9968023, 1.0047264
7: -4.2491102, -2.6610548, -4.2241211, -2.6624391, -0.7338326, 0.7167063
8: -3.3749824, -1.8347774, -3.3512707, -1.8351245, -0.6125581, 0.5954034
9: -12.0794172, -10.4251995, -12.0340996, -10.4266996, -0.7680824, 0.7368430

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2943372, upper bound: 0.2943346
time: 6.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967364, upper bound: 0.2943370
time: 5.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.51 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943357
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967349
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943372
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967345
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2964544
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2988543
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2943372
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2944292, upper bound: 0.2967364
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2943378, upper bound: 0.2943372
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2967369, upper bound: 0.2943336
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2943378, upper bound: 0.2943367
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2967369, upper bound: 0.2943372
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2943372, upper bound: 0.2964554
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2967364, upper bound: 0.2964527
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2943372, upper bound: 0.2943346
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.51
Output dim: 1, lower bound: -0.2967364, upper bound: 0.2943370

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9323978, -7.0598145, -1.0442755, 1.0442760
1: 2.4457912, 3.9116077, 2.4457912, 3.9116077, -0.7839767, 0.7839764
2: -6.5152645, -5.0031300, -6.5152645, -5.0031300, -0.7908618, 0.7908618
3: -11.4202719, -9.5516996, -11.4202719, -9.5516996, -0.9010282, 0.9010282
4: -4.3153510, -2.8075817, -4.3153510, -2.8075817, -0.9396536, 0.9396536
5: -12.3369751, -10.5831194, -12.3369751, -10.5831194, -0.7645340, 0.7645340
6: -10.0451155, -8.0904264, -10.0451155, -8.0904264, -0.9250298, 0.9250298
7: -4.2133045, -2.7129230, -4.2133045, -2.7129230, -0.6756483, 0.6756483
8: -3.2897959, -1.8628788, -3.2897959, -1.8628788, -0.5554231, 0.5554231
9: -12.0044041, -10.4814310, -12.0044041, -10.4814310, -0.6991043, 0.6991043

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944057, upper bound: 0.2943357
time: 4.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944298, upper bound: 0.2943371
time: 4.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9574375, -7.0427332, -1.0550437, 1.0545421
1: 2.4457912, 3.9116077, 2.4158928, 3.9457765, -0.7896037, 0.7951356
2: -6.5152645, -5.0031300, -6.5342789, -4.9874897, -0.7981439, 0.7949076
3: -11.4202719, -9.5516996, -11.4780359, -9.5044737, -0.9200296, 0.9149157
4: -4.3153510, -2.8075817, -4.3626933, -2.7496901, -0.9460371, 0.9562769
5: -12.3369751, -10.5831194, -12.3428955, -10.5804071, -0.7675893, 0.7707762
6: -10.0451155, -8.0904264, -10.0657549, -8.0696087, -0.9377604, 0.9461365
7: -4.2133045, -2.7129230, -4.2381773, -2.6925266, -0.6851747, 0.6829268
8: -3.2897959, -1.8628788, -3.3134413, -1.8440013, -0.5668918, 0.5653759
9: -12.0044041, -10.4814310, -12.0498419, -10.4443741, -0.7105367, 0.7057432

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944057, upper bound: 0.2967319
time: 4.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944298, upper bound: 0.2967369
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9575291, -7.0413122, -8.9323978, -7.0598145, -1.0666447, 1.0557513
1: 2.4301202, 3.9516675, 2.4457912, 3.9116077, -0.8010913, 0.7961979
2: -6.5885110, -4.9888945, -6.5152645, -5.0031300, -0.8083816, 0.8002399
3: -11.4273415, -9.4792500, -11.4202719, -9.5516996, -0.9086492, 0.9358566
4: -4.3275695, -2.7769396, -4.3153510, -2.8075817, -0.9528377, 0.9479160
5: -12.3439207, -10.5368347, -12.3369751, -10.5831194, -0.7716123, 0.7825890
6: -10.0616398, -8.0593510, -10.0451155, -8.0904264, -0.9409823, 0.9430079
7: -4.2235985, -2.6969047, -4.2133045, -2.7129230, -0.6862999, 0.6886724
8: -3.3302660, -1.8539295, -3.2897959, -1.8628788, -0.5721172, 0.5646516
9: -12.0335999, -10.4701977, -12.0044041, -10.4814310, -0.7080691, 0.7105387

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944056, upper bound: 0.2943330
time: 7.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944296, upper bound: 0.2943342
time: 6.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9575291, -7.0413122, -8.9574375, -7.0427332, -1.0700183, 1.0586233
1: 2.4301202, 3.9516675, 2.4158928, 3.9457765, -0.8034793, 0.8020499
2: -6.5885110, -4.9888945, -6.5342789, -4.9874897, -0.8124521, 0.8010740
3: -11.4273415, -9.4792500, -11.4780359, -9.5044737, -0.9227009, 0.9370277
4: -4.3275695, -2.7769396, -4.3626933, -2.7496901, -0.9554183, 0.9596069
5: -12.3439207, -10.5368347, -12.3428955, -10.5804071, -0.7746956, 0.7859411
6: -10.0616398, -8.0593510, -10.0657549, -8.0696087, -0.9529295, 0.9520001
7: -4.2235985, -2.6969047, -4.2381773, -2.6925266, -0.6912844, 0.6896310
8: -3.3302660, -1.8539295, -3.3134413, -1.8440013, -0.5743619, 0.5709336
9: -12.0335999, -10.4701977, -12.0498419, -10.4443741, -0.7139561, 0.7145355

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944056, upper bound: 0.2967306
time: 5.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944296, upper bound: 0.2967339
time: 7.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9575291, -7.0413122, -1.0557518, 1.0666449
1: 2.4457912, 3.9116077, 2.4301202, 3.9516675, -0.7961979, 0.8010911
2: -6.5152645, -5.0031300, -6.5885110, -4.9888945, -0.8002399, 0.8083817
3: -11.4202719, -9.5516996, -11.4273415, -9.4792500, -0.9358566, 0.9086492
4: -4.3153510, -2.8075817, -4.3275695, -2.7769396, -0.9479160, 0.9528377
5: -12.3369751, -10.5831194, -12.3439207, -10.5368347, -0.7825890, 0.7716124
6: -10.0451155, -8.0904264, -10.0616398, -8.0593510, -0.9430079, 0.9409822
7: -4.2133045, -2.7129230, -4.2235985, -2.6969047, -0.6886724, 0.6863000
8: -3.2897959, -1.8628788, -3.3302660, -1.8539295, -0.5646515, 0.5721172
9: -12.0044041, -10.4814310, -12.0335999, -10.4701977, -0.7105389, 0.7080692

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944051, upper bound: 0.2964505
time: 4.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944291, upper bound: 0.2964553
time: 4.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9323978, -7.0598145, -8.9826517, -7.0242376, -1.0591154, 1.0695555
1: 2.4457912, 3.9116077, 2.4002421, 3.9858348, -0.7965190, 0.8089941
2: -6.5152645, -5.0031300, -6.6075225, -4.9732537, -0.8043090, 0.8092154
3: -11.4202719, -9.5516996, -11.4851055, -9.4320507, -0.9421444, 0.9175873
4: -4.3153510, -2.8075817, -4.3749046, -2.7190466, -0.9493665, 0.9656363
5: -12.3369751, -10.5831194, -12.3498621, -10.5341406, -0.7856911, 0.7749670
6: -10.0451155, -8.0904264, -10.0821323, -8.0385342, -0.9436231, 0.9613767
7: -4.2133045, -2.7129230, -4.2484741, -2.6765144, -0.6918906, 0.6890361
8: -3.2897959, -1.8628788, -3.3539162, -1.8350530, -0.5724497, 0.5728462
9: -12.0044041, -10.4814310, -12.0790367, -10.4331493, -0.7193642, 0.7091632

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944051, upper bound: 0.2988505
time: 4.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944291, upper bound: 0.2988540
time: 4.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9874249, -7.0410767, -8.9874249, -7.0410767, -1.0991592, 1.0991595
1: 2.4295378, 3.9733124, 2.4295378, 3.9733124, -0.8283989, 0.8283991
2: -6.6145153, -4.9886618, -6.6145153, -4.9886618, -0.8338931, 0.8338932
3: -11.4273510, -9.4406414, -11.4273510, -9.4406414, -0.9734499, 0.9734499
4: -4.3281822, -2.7685292, -4.3281822, -2.7685292, -0.9642390, 0.9642389
5: -12.3439198, -10.5196209, -12.3439198, -10.5196209, -0.7962382, 0.7962382
6: -10.0847330, -8.0592766, -10.0847330, -8.0592766, -0.9874148, 0.9874148
7: -4.2238207, -2.6814618, -4.2238207, -2.6814618, -0.7110884, 0.7110887
8: -3.3511605, -1.8536549, -3.3511605, -1.8536549, -0.5922900, 0.5922900
9: -12.0337706, -10.4622965, -12.0337706, -10.4622965, -0.7289948, 0.7289950

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965233, upper bound: 0.2943351
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965480, upper bound: 0.2943347
time: 4.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9874249, -7.0410767, -9.0124893, -7.0240126, -1.1025162, 1.1020107
1: 2.4295378, 3.9733124, 2.3996754, 4.0074782, -0.8287179, 0.8342314
2: -6.6145153, -4.9886618, -6.6335292, -4.9730301, -0.8379614, 0.8347272
3: -11.4273510, -9.4406414, -11.4851179, -9.3934479, -0.9797292, 0.9746213
4: -4.3281822, -2.7685292, -4.3754997, -2.7106390, -0.9656885, 0.9758970
5: -12.3439198, -10.5196209, -12.3498611, -10.5169296, -0.7993424, 0.7995991
6: -10.0847330, -8.0592766, -10.1052685, -8.0384579, -0.9880300, 0.9963899
7: -4.2238207, -2.6814618, -4.2486954, -2.6610749, -0.7143199, 0.7120471
8: -3.3511605, -1.8536549, -3.3748045, -1.8347783, -0.5945348, 0.5930123
9: -12.0337706, -10.4622965, -12.0792065, -10.4252529, -0.7349033, 0.7300892

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965233, upper bound: 0.2967305
time: 6.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965480, upper bound: 0.2967338
time: 4.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9578161, -7.0427113, -1.0573475, 1.0573478
1: 2.4158239, 3.9457893, 2.4158239, 3.9457893, -0.7914399, 0.7914399
2: -6.5345716, -4.9874301, -6.5345716, -4.9874301, -0.7990620, 0.7990620
3: -11.4782887, -9.5044403, -11.4782887, -9.5044403, -0.9087915, 0.9087914
4: -4.3628030, -2.7489786, -4.3628030, -2.7489786, -0.9578516, 0.9578519
5: -12.3429804, -10.5802021, -12.3429804, -10.5802021, -0.7774388, 0.7774389
6: -10.0658302, -8.0694809, -10.0658302, -8.0694809, -0.9380934, 0.9380933
7: -4.2385912, -2.6925077, -4.2385912, -2.6925077, -0.6846280, 0.6846279
8: -3.3136191, -1.8440018, -3.3136191, -1.8440018, -0.5588044, 0.5588044
9: -12.0500526, -10.4443245, -12.0500526, -10.4443245, -0.7083757, 0.7083757

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967097, upper bound: 0.2943332
time: 7.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967370, upper bound: 0.2943360
time: 4.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9830284, -7.0242138, -8.9578161, -7.0427113, -1.0765452, 1.0656726
1: 2.4001737, 3.9858458, 2.4158239, 3.9457893, -0.8085490, 0.8025668
2: -6.6078167, -4.9731927, -6.5345716, -4.9874301, -0.8154435, 0.8073014
3: -11.4853611, -9.4320202, -11.4782887, -9.5044403, -0.9164135, 0.9436526
4: -4.3750114, -2.7183342, -4.3628030, -2.7489786, -0.9704305, 0.9643955
5: -12.3499432, -10.5339375, -12.3429804, -10.5802021, -0.7808217, 0.7918050
6: -10.0822086, -8.0384064, -10.0658302, -8.0694809, -0.9542139, 0.9530783
7: -4.2488880, -2.6764944, -4.2385912, -2.6925077, -0.6938566, 0.6944654
8: -3.3540945, -1.8350530, -3.3136191, -1.8440018, -0.5758340, 0.5680330
9: -12.0792465, -10.4330978, -12.0500526, -10.4443245, -0.7166338, 0.7198610

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988265, upper bound: 0.2943325
time: 6.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988520, upper bound: 0.2943335
time: 4.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9574375, -7.0427332, -8.9575291, -7.0413122, -1.0586233, 1.0700183
1: 2.4158928, 3.9457765, 2.4301202, 3.9516675, -0.8020499, 0.8034794
2: -6.5342789, -4.9874897, -6.5885110, -4.9888945, -0.8010741, 0.8124520
3: -11.4780359, -9.5044737, -11.4273415, -9.4792500, -0.9370277, 0.9227009
4: -4.3626933, -2.7496901, -4.3275695, -2.7769396, -0.9596069, 0.9554183
5: -12.3428955, -10.5804071, -12.3439207, -10.5368347, -0.7859411, 0.7746956
6: -10.0657549, -8.0696087, -10.0616398, -8.0593510, -0.9520001, 0.9529293
7: -4.2381773, -2.6925266, -4.2235985, -2.6969047, -0.6896309, 0.6912844
8: -3.3134413, -1.8440013, -3.3302660, -1.8539295, -0.5709337, 0.5743620
9: -12.0498419, -10.4443741, -12.0335999, -10.4701977, -0.7145354, 0.7139562

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2943131, upper bound: 0.2943356
time: 4.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943371, upper bound: 0.2964535
time: 6.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9578161, -7.0427113, -8.9830284, -7.0242138, -1.0656729, 1.0765460
1: 2.4158239, 3.9457893, 2.4001737, 3.9858458, -0.8025669, 0.8085490
2: -6.5345716, -4.9874301, -6.6078167, -4.9731927, -0.8073015, 0.8154435
3: -11.4782887, -9.5044403, -11.4853611, -9.4320202, -0.9436529, 0.9164134
4: -4.3628030, -2.7489786, -4.3750114, -2.7183342, -0.9643954, 0.9704304
5: -12.3429804, -10.5802021, -12.3499432, -10.5339375, -0.7918048, 0.7808216
6: -10.0658302, -8.0694809, -10.0822086, -8.0384064, -0.9530783, 0.9542136
7: -4.2385912, -2.6925077, -4.2488880, -2.6764944, -0.6944655, 0.6938565
8: -3.3136191, -1.8440018, -3.3540945, -1.8350530, -0.5680331, 0.5758340
9: -12.0500526, -10.4443245, -12.0792465, -10.4330978, -0.7198610, 0.7166339

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2943131, upper bound: 0.2964526
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2964545
time: 4.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0128641, -7.0239873, -9.0128641, -7.0239873, -1.1090088, 1.1090086
1: 2.3996077, 4.0074911, 2.3996077, 4.0074911, -0.8347470, 0.8347471
2: -6.6338215, -4.9729686, -6.6338215, -4.9729686, -0.8409543, 0.8409544
3: -11.4853725, -9.3934155, -11.4853725, -9.3934155, -0.9812376, 0.9812374
4: -4.3756084, -2.7099276, -4.3756084, -2.7099276, -0.9806896, 0.9806896
5: -12.3499451, -10.5167255, -12.3499451, -10.5167255, -0.8054702, 0.8054703
6: -10.1053467, -8.0383301, -10.1053467, -8.0383301, -0.9974756, 0.9974754
7: -4.2491102, -2.6610548, -4.2491102, -2.6610548, -0.7168947, 0.7168947
8: -3.3749824, -1.8347774, -3.3749824, -1.8347774, -0.5960014, 0.5960014
9: -12.0794172, -10.4251995, -12.0794172, -10.4251995, -0.7375844, 0.7375845

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988273, upper bound: 0.2943350
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988543, upper bound: 0.2943338
time: 4.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.83 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944057, upper bound: 0.2943357
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944298, upper bound: 0.2943371
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944057, upper bound: 0.2967319
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944298, upper bound: 0.2967369
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944056, upper bound: 0.2943330
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944296, upper bound: 0.2943342
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944056, upper bound: 0.2967306
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944296, upper bound: 0.2967339
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944051, upper bound: 0.2964505
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944291, upper bound: 0.2964553
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944051, upper bound: 0.2988505
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2944291, upper bound: 0.2988540
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2965233, upper bound: 0.2943351
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2965480, upper bound: 0.2943347
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2965233, upper bound: 0.2967305
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2965480, upper bound: 0.2967338
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2967097, upper bound: 0.2943332
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2967370, upper bound: 0.2943360
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2988265, upper bound: 0.2943325
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2988520, upper bound: 0.2943335
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2943131, upper bound: 0.2943356
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2943371, upper bound: 0.2964535
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2943131, upper bound: 0.2964526
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2967363, upper bound: 0.2964545
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2988273, upper bound: 0.2943350
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.83
Output dim: 1, lower bound: -0.2988543, upper bound: 0.2943338

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9240828, -7.0637317, -8.9255924, -7.0631247, -1.0214577, 1.0225446
1: 2.4498925, 3.9088790, 2.4492919, 3.9093976, -0.7744941, 0.7752818
2: -6.5107188, -5.0191026, -6.5115681, -5.0164256, -0.7713730, 0.7693448
3: -11.4016037, -9.5619602, -11.4049587, -9.5600605, -0.8754592, 0.8774167
4: -4.3047762, -2.8126082, -4.3066440, -2.8116543, -0.9232566, 0.9243433
5: -12.3255329, -10.5903902, -12.3274641, -10.5890141, -0.7502379, 0.7517405
6: -10.0291052, -8.0937738, -10.0319386, -8.0933084, -0.9020519, 0.9050843
7: -4.2038727, -2.7137659, -4.2055502, -2.7136374, -0.6645181, 0.6661893
8: -3.2858357, -1.8793821, -3.2865744, -1.8766046, -0.5370343, 0.5348504
9: -12.0012283, -10.4942122, -12.0016689, -10.4919205, -0.6814275, 0.6795616

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944059, upper bound: 0.2944045
time: 5.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944059, upper bound: 0.2944256
time: 6.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9422169, -7.0481277, -8.9323902, -7.0598211, -1.0473022, 1.0597794
1: 2.4368093, 3.9354696, 2.4457951, 3.9116056, -0.7978289, 0.8017159
2: -6.5557747, -4.9985342, -6.5152612, -5.0031428, -0.8050929, 0.7858150
3: -11.4214487, -9.5013666, -11.4202595, -9.5517082, -0.8910694, 0.9291780
4: -4.3229990, -2.7949777, -4.3153429, -2.8075855, -0.9453619, 0.9460416
5: -12.3380241, -10.5531912, -12.3369694, -10.5831261, -0.7642319, 0.7865109
6: -10.0578709, -8.0456705, -10.0450878, -8.0904284, -0.9276903, 0.9545653
7: -4.2170358, -2.6908677, -4.2132969, -2.7129235, -0.6724765, 0.6876867
8: -3.3242517, -1.8591743, -3.2897911, -1.8628988, -0.5690017, 0.5468393
9: -12.0404148, -10.4783115, -12.0044012, -10.4814529, -0.7087165, 0.6952695

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944279, upper bound: 0.2944037
time: 4.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944279, upper bound: 0.2944059
time: 4.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9240828, -7.0637317, -8.9506388, -7.0460410, -1.0321345, 1.0327227
1: 2.4498925, 3.9088790, 2.4193761, 3.9435675, -0.7802001, 0.7863653
2: -6.5107188, -5.0191026, -6.5305815, -5.0007725, -0.7781370, 0.7734456
3: -11.4016037, -9.5619602, -11.4627237, -9.5128288, -0.8944499, 0.8903759
4: -4.3047762, -2.8126082, -4.3540120, -2.7537642, -0.9295467, 0.9406582
5: -12.3255329, -10.5903902, -12.3333492, -10.5863180, -0.7532816, 0.7579675
6: -10.0291052, -8.0937738, -10.0525942, -8.0724888, -0.9141765, 0.9251432
7: -4.2038727, -2.7137659, -4.2304239, -2.6932452, -0.6739913, 0.6731194
8: -3.2858357, -1.8793821, -3.3102231, -1.8577271, -0.5479305, 0.5449146
9: -12.0012283, -10.4942122, -12.0471058, -10.4548531, -0.6921694, 0.6861610

Time for backsubstitution: 14.41 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8209632635116577
rel_dist={1: [-0.29885889706970215, 0.2988555407171436]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1656.18 seconds
