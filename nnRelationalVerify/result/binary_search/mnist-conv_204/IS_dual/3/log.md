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
execution time: IAR + LP analysis = 15.08 + 31.69 = 46.77 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3553.23 seconds, max iter: 100)

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
Binary search time: 147.31 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 3405.92 seconds

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
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5112008
time: 5.08 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5181364
time: 6.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.33 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 11.33
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5112008
IS_B2, status: Status.UNKNOWN, split count: 1, time: 11.33
Output dim: 1, lower bound: -0.5181404, upper bound: 0.5181364

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.9362888, -7.0519896, -8.9359207, -7.0585661, -1.2190421, 1.2259290
1: 2.4288263, 3.9131081, 2.4402661, 3.9130864, -0.9198203, 0.9081159
2: -6.5179396, -4.9858332, -6.5177808, -4.9914556, -0.9231153, 0.9290212
3: -11.4268990, -9.5287027, -11.4268007, -9.5475321, -1.0785265, 1.0976034
4: -4.3398786, -2.8067675, -4.3217554, -2.8069339, -1.0747352, 1.0558739
5: -12.3429470, -10.5800352, -12.3428230, -10.5806141, -0.9366270, 0.9370220
6: -10.0559959, -8.0892534, -10.0495138, -8.0893326, -1.1135962, 1.1068571
7: -4.2140799, -2.7035518, -4.2139525, -2.7113905, -0.8052243, 0.8133000
8: -3.2912946, -1.8497934, -3.2912493, -1.8574257, -0.7022642, 0.7099417
9: -12.0049200, -10.4607239, -12.0047808, -10.4753914, -0.8503437, 0.8652500

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5111972
time: 4.03 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181307, upper bound: 0.5111945
time: 5.04 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.9367981, -7.0426021, -8.9613400, -7.0414629, -1.2325575, 1.2509327
1: 2.4124551, 3.9131386, 2.4103026, 3.9472642, -0.9475803, 0.9203613
2: -6.5181546, -4.9778090, -6.5370874, -4.9757576, -0.9331372, 0.9442365
3: -11.4270363, -9.5018044, -11.4848194, -9.5002756, -1.0933900, 1.1411413
4: -4.3657713, -2.8065424, -4.3692021, -2.7483325, -1.1126878, 1.0811603
5: -12.3431263, -10.5792007, -12.3488350, -10.5776911, -0.9414518, 0.9475729
6: -10.0652590, -8.0891495, -10.0702209, -8.0683899, -1.1388216, 1.1202728
7: -4.2142544, -2.6923594, -4.2392397, -2.6909769, -0.8170778, 0.8353376
8: -3.2913561, -1.8388929, -3.3150744, -1.8385506, -0.7082822, 0.7330920
9: -12.0051107, -10.4397821, -12.0504303, -10.4382887, -0.8651667, 0.8973836

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5734
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5181331
time: 3.93 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181329, upper bound: 0.5181302
time: 6.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.51 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.51
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5111972
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.51
Output dim: 1, lower bound: -0.5181307, upper bound: 0.5111945
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 24.51
Output dim: 1, lower bound: -0.5118730, upper bound: 0.5181331
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.51
Output dim: 1, lower bound: -0.5181329, upper bound: 0.5181302

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -8.9327660, -7.0532417, -8.9348497, -7.0589561, -1.2150443, 1.2224655
1: 2.4343529, 3.9116294, 2.4420075, 3.9125435, -0.9118909, 0.9049358
2: -6.5154247, -4.9975090, -6.5170116, -4.9951348, -0.9174378, 0.9167070
3: -11.4203701, -9.5328674, -11.4247446, -9.5488081, -1.0697000, 1.0919033
4: -4.3334756, -2.8074155, -4.3197384, -2.8071301, -1.0657389, 1.0519371
5: -12.3371010, -10.5825405, -12.3409834, -10.5813866, -0.9292558, 0.9323452
6: -10.0515938, -8.0903482, -10.0481071, -8.0896759, -1.1079841, 1.1030843
7: -4.2134323, -2.7050829, -4.2137489, -2.7118526, -0.8034439, 0.8117309
8: -3.2898402, -1.8552446, -3.2908077, -1.8591471, -0.6985228, 0.7025557
9: -12.0045433, -10.4667616, -12.0046654, -10.4773054, -0.8470178, 0.8570645

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111971
time: 4.01 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111953
time: 4.10 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.9795675, -7.0345540, -8.9359150, -7.0585661, -1.2599988, 1.2351935
1: 2.4182420, 3.9675157, 2.4402795, 3.9130833, -0.9367100, 0.9374490
2: -6.6076875, -4.9830785, -6.5177765, -4.9914780, -0.9516468, 0.9292519
3: -11.4274464, -9.4322033, -11.4267817, -9.5475397, -1.0810673, 1.1502874
4: -4.3461485, -2.7706292, -4.3217416, -2.8069355, -1.0804774, 1.0690784
5: -12.3440428, -10.5239925, -12.3428011, -10.5806179, -0.9360842, 0.9595922
6: -10.0845070, -8.0592175, -10.0495062, -8.0893345, -1.1564484, 1.1252952
7: -4.2238836, -2.6777494, -4.2139511, -2.7113938, -0.8175923, 0.8356638
8: -3.3455758, -1.8460617, -3.2912474, -1.8574352, -0.7289613, 0.7183566
9: -12.0338669, -10.4497557, -12.0047808, -10.4754047, -0.8614815, 0.8811318

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109319, upper bound: 0.5111962
time: 4.17 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109319, upper bound: 0.5111964
time: 4.16 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -8.9332771, -7.0438547, -8.9602642, -7.0418530, -1.2285624, 1.2474370
1: 2.4179807, 3.9116602, 2.4120431, 3.9467220, -0.9396249, 0.9171736
2: -6.5156393, -4.9894838, -6.5363197, -4.9794364, -0.9274571, 0.9318428
3: -11.4205065, -9.5059700, -11.4827642, -9.5015516, -1.0845630, 1.1350927
4: -4.3593702, -2.8071899, -4.3671870, -2.7485275, -1.1036355, 1.0772235
5: -12.3372784, -10.5817070, -12.3469954, -10.5784674, -0.9340802, 0.9426574
6: -10.0608521, -8.0902424, -10.0688152, -8.0687304, -1.1330557, 1.1165259
7: -4.2136049, -2.6938910, -4.2390366, -2.6914389, -0.8152972, 0.8337436
8: -3.2899027, -1.8443446, -3.3146319, -1.8402672, -0.7045413, 0.7256860
9: -12.0047331, -10.4458199, -12.0503111, -10.4402027, -0.8618400, 0.8892280

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5181334
time: 3.94 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5109302
time: 4.53 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.9801044, -7.0251665, -8.9613323, -7.0414639, -1.2708130, 1.2534437
1: 2.4018750, 3.9675465, 2.4103165, 3.9472609, -0.9575286, 0.9486752
2: -6.6079025, -4.9750571, -6.5370841, -4.9757805, -0.9607623, 0.9445815
3: -11.4275837, -9.4053030, -11.4848003, -9.5002832, -1.0959308, 1.1852708
4: -4.3720369, -2.7704003, -4.3691893, -2.7483327, -1.1175208, 1.0929831
5: -12.3442173, -10.5231628, -12.3488178, -10.5776968, -0.9409072, 0.9659523
6: -10.0937443, -8.0591125, -10.0702114, -8.0683918, -1.1704712, 1.1373634
7: -4.2240567, -2.6665609, -4.2392406, -2.6909800, -0.8273264, 0.8523295
8: -3.3456411, -1.8351607, -3.3150721, -1.8385572, -0.7354132, 0.7348911
9: -12.0340595, -10.4288197, -12.0504274, -10.4383011, -0.8757489, 0.9096477

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109319, upper bound: 0.5181335
time: 4.42 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109295, upper bound: 0.5109311
time: 4.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.48 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111971
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111953
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5109319, upper bound: 0.5111962
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5109319, upper bound: 0.5111964
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5181334
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5109302
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5109319, upper bound: 0.5181335
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.48
Output dim: 1, lower bound: -0.5109295, upper bound: 0.5109311

## BFS IS instance: IS_B1_A1_A1

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
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5049370
time: 4.14 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111971
time: 4.03 seconds

## BFS IS instance: IS_B1_A1_A2

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
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5049351
time: 4.64 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111953
time: 4.13 seconds

## BFS IS instance: IS_B1_A2_A1

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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109316, upper bound: 0.5049355
time: 4.73 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109325, upper bound: 0.5049355
time: 4.52 seconds

## BFS IS instance: IS_B1_A2_A2

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

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109316, upper bound: 0.5049354
time: 4.49 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109325, upper bound: 0.5049342
time: 4.45 seconds

## BFS IS instance: IS_B2_A1_A1

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

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5118734
time: 4.06 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5181334
time: 3.98 seconds

## BFS IS instance: IS_B2_A1_A2

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

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5046716
time: 4.20 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5109302
time: 4.61 seconds

## BFS IS instance: IS_B2_A2_A1

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

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109292, upper bound: 0.5118719
time: 4.37 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109301, upper bound: 0.5118719
time: 4.53 seconds

## BFS IS instance: IS_B2_A2_A2

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

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109292, upper bound: 0.5046699
time: 4.60 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5109301, upper bound: 0.5046698
time: 4.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.54 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5049370
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111971
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5049351
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5111953
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109316, upper bound: 0.5049355
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109325, upper bound: 0.5049355
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109316, upper bound: 0.5049354
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109325, upper bound: 0.5049342
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5118734
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5181334
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5046716
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5046714, upper bound: 0.5109302
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109292, upper bound: 0.5118719
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109301, upper bound: 0.5118719
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109292, upper bound: 0.5046699
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.54
Output dim: 1, lower bound: -0.5109301, upper bound: 0.5046698

## BFS IS instance: IS_B1_A1_A1_B1

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
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043962, upper bound: 0.5049363
time: 4.20 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5049337
time: 4.12 seconds

## BFS IS instance: IS_B1_A1_A1_B2

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
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049346, upper bound: 0.5106854
time: 4.01 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049346, upper bound: 0.5111946
time: 3.83 seconds

## BFS IS instance: IS_B1_A1_A2_B1

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

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5113282, upper bound: 0.5049326
time: 3.99 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118685, upper bound: 0.5049353
time: 4.04 seconds

## BFS IS instance: IS_B1_A1_A2_B2

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

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118711, upper bound: 0.5106850
time: 3.83 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5118685, upper bound: 0.5111963
time: 4.12 seconds

## BFS IS instance: IS_B1_A2_A1_B1

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

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106868, upper bound: 0.5049320
time: 3.98 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111930, upper bound: 0.5049322
time: 4.14 seconds

## BFS IS instance: IS_B1_A2_A1_B2

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

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106876, upper bound: 0.5049320
time: 3.99 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111939, upper bound: 0.5049318
time: 4.15 seconds

## BFS IS instance: IS_B1_A2_A2_B1

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

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5176175, upper bound: 0.5049310
time: 4.13 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181282, upper bound: 0.5049346
time: 4.70 seconds

## BFS IS instance: IS_B1_A2_A2_B2

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

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181291, upper bound: 0.5043962
time: 5.06 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5181291, upper bound: 0.5049314
time: 4.46 seconds

## BFS IS instance: IS_B2_A1_A1_B1

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

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_B2_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049321, upper bound: 0.5113281
time: 4.17 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049346, upper bound: 0.5118707
time: 3.84 seconds

## BFS IS instance: IS_B2_A1_A1_B2

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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_B2_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049346, upper bound: 0.5176187
time: 3.93 seconds

## Relational analysis of IS_B2_A1_A1_B2_B2

### Relational analysis result of IS_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5049346, upper bound: 0.5181309
time: 3.93 seconds

## BFS IS instance: IS_B2_A1_A2_B1

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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043209, upper bound: 0.5042799
time: 4.23 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111906, upper bound: 0.5042828
time: 4.25 seconds

## BFS IS instance: IS_B2_A1_A2_B2

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

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5043209, upper bound: 0.5105414
time: 4.31 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111906, upper bound: 0.5105443
time: 4.44 seconds

## BFS IS instance: IS_B2_A2_A1_B1

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

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5106842, upper bound: 0.5118684
time: 4.14 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.5111930, upper bound: 0.5118679
time: 4.80 seconds

## BFS IS instance: IS_B2_A2_A1_B2

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

Time for backsubstitution: 14.44 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.9366202354431152
rel_dist={1: [-0.5181458851188188, 0.518142788570696]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5734
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5734

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965503, upper bound: 0.2988561
time: 4.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988571, upper bound: 0.2988548
time: 4.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.94
Output dim: 1, lower bound: -0.2965503, upper bound: 0.2988561
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.94
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

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

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
time: 6.65 seconds

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

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

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
- Time for IS candidates: 23.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.27
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2967341
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.27
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2988550
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.27
Output dim: 1, lower bound: -0.2988551, upper bound: 0.2967331
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.27
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

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2943351
time: 6.55 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2967340
time: 7.43 seconds

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
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2964536
time: 6.15 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2988550
time: 5.18 seconds

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

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988002, upper bound: 0.2941924
time: 6.25 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2986032, upper bound: 0.2964840
time: 6.55 seconds

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

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988002, upper bound: 0.2963112
time: 5.90 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2986032, upper bound: 0.2986028
time: 4.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.66 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2943351
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2967340
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2964536
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2965483, upper bound: 0.2988550
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2988002, upper bound: 0.2941924
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2986032, upper bound: 0.2964840
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2988002, upper bound: 0.2963112
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 1, lower bound: -0.2986032, upper bound: 0.2986028

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.9333134, -7.0595007, -8.9323978, -7.0598145, -1.0451033, 1.0448875
1: 2.4444103, 3.9119303, 2.4457912, 3.9116077, -0.7859069, 0.7841741
2: -6.5159121, -5.0002294, -6.5152645, -5.0031300, -0.7913833, 0.7937968
3: -11.4218979, -9.5506229, -11.4202719, -9.5516996, -0.9029813, 0.9018326
4: -4.3169527, -2.8074145, -4.3153510, -2.8075817, -0.9418111, 0.9399498
5: -12.3384361, -10.5824785, -12.3369751, -10.5831194, -0.7661729, 0.7652059
6: -10.0461960, -8.0901537, -10.0451155, -8.0904264, -0.9262762, 0.9255972
7: -4.2134676, -2.7125211, -4.2133045, -2.7129230, -0.6760105, 0.6759390
8: -3.2901754, -1.8615208, -3.2897959, -1.8628788, -0.5558333, 0.5571586
9: -12.0044994, -10.4799328, -12.0044041, -10.4814310, -0.6993160, 0.7011008

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943357
time: 5.71 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943372
time: 5.54 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.9333134, -7.0595007, -8.9574375, -7.0427332, -1.0558720, 1.0551255
1: 2.4444103, 3.9119303, 2.4158928, 3.9457765, -0.7912825, 0.7953464
2: -6.5159121, -5.0002294, -6.5342789, -4.9874897, -0.7987144, 0.7974259
3: -11.4218979, -9.5506229, -11.4780359, -9.5044737, -0.9217064, 0.9157139
4: -4.3169527, -2.8074145, -4.3626933, -2.7496901, -0.9479330, 0.9565589
5: -12.3384361, -10.5824785, -12.3428955, -10.5804071, -0.7692282, 0.7714480
6: -10.0461960, -8.0901537, -10.0657549, -8.0696087, -0.9389892, 0.9467015
7: -4.2134676, -2.7125211, -4.2381773, -2.6925266, -0.6855073, 0.6832170
8: -3.2901754, -1.8615208, -3.3134413, -1.8440013, -0.5672966, 0.5668729
9: -12.0044994, -10.4799328, -12.0498419, -10.4443741, -0.7107468, 0.7074950

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967349
time: 5.84 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967345
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.9359131, -7.0585666, -8.9678726, -7.0411921, -1.0593295, 1.0785959
1: 2.4402864, 3.9130826, 2.4298801, 3.9592237, -0.8099122, 0.8047371
2: -6.5177746, -4.9914856, -6.5975933, -4.9887533, -0.8013318, 0.8258988
3: -11.4267712, -9.5475435, -11.4273453, -9.4656868, -0.9556000, 0.9114815
4: -4.3217344, -2.8069353, -4.3278055, -2.7740059, -0.9584945, 0.9559891
5: -12.3427925, -10.5806208, -12.3439198, -10.5309334, -0.7923298, 0.7713135
6: -10.0495014, -8.0893383, -10.0689545, -8.0593214, -0.9480844, 0.9597912
7: -4.2139521, -2.7113943, -4.2236757, -2.6914544, -0.6954472, 0.6893880
8: -3.2912445, -1.8574386, -3.3375540, -1.8537483, -0.5696007, 0.5836774
9: -12.0047817, -10.4754105, -12.0336685, -10.4674234, -0.7186058, 0.7160058

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965235, upper bound: 0.2964532
time: 4.40 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965482, upper bound: 0.2964535
time: 4.43 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.9359131, -7.0585666, -8.9929733, -7.0241203, -1.0626912, 1.0814655
1: 2.4402864, 3.9130826, 2.4000075, 3.9933894, -0.8102334, 0.8105712
2: -6.5177746, -4.9914856, -6.6166067, -4.9731164, -0.8089225, 0.8267332
3: -11.4267712, -9.5475435, -11.4851084, -9.4184885, -0.9618850, 0.9210327
4: -4.3217344, -2.8069353, -4.3751335, -2.7161140, -0.9599448, 0.9676533
5: -12.3427925, -10.5806208, -12.3498611, -10.5282431, -0.7954397, 0.7775675
6: -10.0495014, -8.0893383, -10.0895014, -8.0385046, -0.9486990, 0.9735725
7: -4.2139521, -2.7113943, -4.2485495, -2.6710651, -0.6986651, 0.6903464
8: -3.2912445, -1.8574386, -3.3612027, -1.8348742, -0.5744534, 0.5844035
9: -12.0047817, -10.4754105, -12.0791035, -10.4303761, -0.7245334, 0.7170998

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965235, upper bound: 0.2988501
time: 4.34 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965482, upper bound: 0.2988530
time: 4.48 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.9586420, -7.0444469, -8.9330759, -7.0464821, -1.0726767, 1.0556030
1: 2.4147024, 3.9461107, 2.4182973, 3.9110363, -0.7922323, 0.8176012
2: -6.5352011, -4.9866276, -6.5156131, -4.9922833, -0.8092479, 0.8010197
3: -11.4799061, -9.5044384, -11.4204941, -9.5074329, -0.9618692, 0.9081833
4: -4.3600826, -2.7488141, -4.3536558, -2.8071930, -0.9575137, 0.9892941
5: -12.3444395, -10.5801554, -12.3370571, -10.5825748, -0.7758487, 0.7700305
6: -10.0663738, -8.0692158, -10.0601902, -8.0909300, -0.9353247, 0.9533422
7: -4.2387342, -2.6921649, -4.2126069, -2.6939704, -0.7041430, 0.6836468
8: -3.3139853, -1.8437161, -3.2898531, -1.8457222, -0.5838486, 0.5598165
9: -12.0501451, -10.4445601, -12.0047302, -10.4481449, -0.7414527, 0.7095910

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2966816, upper bound: 0.2941923
time: 4.69 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2966816, upper bound: 0.2941951
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.9584990, -7.0424051, -8.9432325, -7.0427103, -1.0751848, 1.0507219
1: 2.4144559, 3.9445105, 2.4171422, 3.9116545, -0.7923374, 0.8207998
2: -6.5352201, -4.9848323, -6.5244293, -4.9885902, -0.8112730, 0.8022528
3: -11.4799175, -9.5036058, -11.4248838, -9.5058250, -0.9625874, 0.9116369
4: -4.3639231, -2.7488108, -4.3614058, -2.7887702, -0.9618433, 0.9953430
5: -12.3438597, -10.5795679, -12.3400822, -10.5808411, -0.7765887, 0.7701035
6: -10.0668621, -8.0709658, -10.0631151, -8.0902414, -0.9375978, 0.9604986
7: -4.2362256, -2.6921239, -4.2136378, -2.6942363, -0.7058556, 0.6835893
8: -3.3139091, -1.8426566, -3.2943234, -1.8443551, -0.5840601, 0.5625411
9: -12.0501413, -10.4430857, -12.0119762, -10.4450111, -0.7433598, 0.7110296

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964846, upper bound: 0.2964843
time: 4.69 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964846, upper bound: 0.2964842
time: 4.45 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.9612389, -7.0435143, -8.9685831, -7.0278592, -1.0795116, 1.0833874
1: 2.4105787, 3.9472585, 2.4023914, 3.9586523, -0.8152118, 0.8328490
2: -6.5370631, -4.9778867, -6.5979409, -4.9779119, -0.8194561, 0.8303268
3: -11.4847813, -9.5013609, -11.4275675, -9.4214191, -1.0020609, 0.9178380
4: -4.3648615, -2.7483361, -4.3661003, -2.7736154, -0.9695158, 1.0004237
5: -12.3488026, -10.5782948, -12.3439903, -10.5303974, -0.7984793, 0.7761298
6: -10.0696697, -8.0683975, -10.0839691, -8.0598297, -0.9556909, 0.9801888
7: -4.2392187, -2.6910398, -4.2229791, -2.6725061, -0.7172879, 0.6938987
8: -3.3150578, -1.8396344, -3.3376169, -1.8365936, -0.5910047, 0.5855127
9: -12.0504274, -10.4400434, -12.0339947, -10.4341440, -0.7552044, 0.7218937

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 820

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2987969, upper bound: 0.2962892
time: 4.70 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2988001, upper bound: 0.2963114
time: 7.28 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.9610996, -7.0414753, -8.9787683, -7.0240932, -1.0820055, 1.0867007
1: 2.4103322, 3.9456582, 2.4012597, 3.9592690, -0.8155237, 0.8360252
2: -6.5370803, -4.9760904, -6.6067548, -4.9742265, -0.8214732, 0.8323300
3: -11.4847908, -9.5005245, -11.4319582, -9.4198265, -1.0027652, 0.9212843
4: -4.3687019, -2.7483330, -4.3738232, -2.7551937, -0.9738488, 1.0064306
5: -12.3482227, -10.5777092, -12.3470182, -10.5286713, -0.7992233, 0.7762067
6: -10.0701580, -8.0701494, -10.0868492, -8.0591373, -0.9562750, 0.9873898
7: -4.2367105, -2.6909966, -4.2240095, -2.6727748, -0.7190143, 0.6941798
8: -3.3149824, -1.8385725, -3.3420849, -1.8352280, -0.5912166, 0.5867910
9: -12.0504217, -10.4385681, -12.0412388, -10.4310217, -0.7571571, 0.7234523

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5734
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2985785, upper bound: 0.2986001
time: 5.04 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2986032, upper bound: 0.2986031
time: 5.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.84 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943357
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2943372
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967349
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967345
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2965235, upper bound: 0.2964532
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2965482, upper bound: 0.2964535
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2965235, upper bound: 0.2988501
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2965482, upper bound: 0.2988530
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2966816, upper bound: 0.2941923
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2966816, upper bound: 0.2941951
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2964846, upper bound: 0.2964843
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2964846, upper bound: 0.2964842
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2987969, upper bound: 0.2962892
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2988001, upper bound: 0.2963114
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2985785, upper bound: 0.2986001
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.84
Output dim: 1, lower bound: -0.2986032, upper bound: 0.2986031

## BFS IS instance: IS_A1_B1_B1_A1

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

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944058, upper bound: 0.2944257
time: 4.46 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944298, upper bound: 0.2944277
time: 4.60 seconds

## BFS IS instance: IS_A1_B1_B1_A2

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

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944058, upper bound: 0.2944254
time: 5.76 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944298, upper bound: 0.2944261
time: 6.35 seconds

## BFS IS instance: IS_A1_B1_B2_A1

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

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944276, upper bound: 0.2967077
time: 4.82 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944297, upper bound: 0.2967352
time: 4.23 seconds

## BFS IS instance: IS_A1_B1_B2_A2

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

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 6220
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944056, upper bound: 0.2967296
time: 7.08 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944296, upper bound: 0.2967324
time: 4.52 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9276381, -7.0624399, -8.9599237, -7.0444250, -1.0361929, 1.0557773
1: 2.4443979, 3.9104722, 2.4337745, 3.9562464, -0.7998457, 0.7956444
2: -6.5129547, -5.0074420, -6.5929770, -5.0026159, -0.7814524, 0.8037549
3: -11.4081125, -9.5576782, -11.4120474, -9.4758835, -0.9280059, 0.8877884
4: -4.3111444, -2.8119528, -4.3189058, -2.7783794, -0.9417838, 0.9400903
5: -12.3313522, -10.5877848, -12.3341980, -10.5376673, -0.7770936, 0.7583518
6: -10.0334396, -8.0927525, -10.0550756, -8.0626879, -0.9242048, 0.9387761
7: -4.2045126, -2.7122257, -4.2159519, -2.6930344, -0.6834426, 0.6795779
8: -3.2873034, -1.8739486, -3.3336062, -1.8681974, -0.5503275, 0.5626504
9: -12.0015955, -10.4882154, -12.0305052, -10.4782114, -0.6998156, 0.6961745

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965237, upper bound: 0.2965235
time: 4.87 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965237, upper bound: 0.2965454
time: 4.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9459009, -7.0456133, -8.9691763, -7.0411868, -1.0603518, 1.0890865
1: 2.4308085, 3.9383790, 2.4298618, 3.9601827, -0.8200634, 0.8190453
2: -6.5584602, -4.9863763, -6.5987430, -4.9887705, -0.8163469, 0.8221707
3: -11.4280176, -9.4958572, -11.4273252, -9.4639902, -0.9472241, 0.9364228
4: -4.3297524, -2.7941709, -4.3278170, -2.7736378, -0.9649153, 0.9575257
5: -12.3438921, -10.5488529, -12.3439140, -10.5302010, -0.7924918, 0.7956500
6: -10.0624084, -8.0437260, -10.0699806, -8.0593262, -0.9507136, 0.9844892
7: -4.2180338, -2.6888964, -4.2236743, -2.6907718, -0.6929297, 0.6955177
8: -3.3258572, -1.8531594, -3.3384776, -1.8537645, -0.5767689, 0.5761234
9: -12.0421772, -10.4722948, -12.0336714, -10.4671011, -0.7238796, 0.7084183

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944293, upper bound: 0.2965475
time: 4.26 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944269, upper bound: 0.2944293
time: 4.92 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9276381, -7.0624399, -8.9850330, -7.0273514, -1.0395522, 1.0586574
1: 2.4443979, 3.9104722, 2.4038858, 3.9904122, -0.8001657, 0.8014715
2: -6.5129547, -5.0074420, -6.6119900, -4.9869661, -0.7884783, 0.8045889
3: -11.4081125, -9.5576782, -11.4698124, -9.4286804, -0.9342949, 0.8964461
4: -4.3111444, -2.8119528, -4.3662586, -2.7204871, -0.9432340, 0.9517891
5: -12.3313522, -10.5877848, -12.3401089, -10.5349922, -0.7802093, 0.7645917
6: -10.0334396, -8.0927525, -10.0756474, -8.0418682, -0.9248199, 0.9519794
7: -4.2045126, -2.7122257, -4.2408276, -2.6726470, -0.6866596, 0.6805369
8: -3.2873034, -1.8739486, -3.3572540, -1.8493195, -0.5548418, 0.5633806
9: -12.0015955, -10.4882154, -12.0759411, -10.4411526, -0.7057415, 0.6972691

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965235, upper bound: 0.2988292
time: 4.28 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2965235, upper bound: 0.2988513
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9459009, -7.0456133, -8.9942760, -7.0241170, -1.0637131, 1.0919549
1: 2.4308085, 3.9383790, 2.3999903, 3.9943485, -0.8203847, 0.8248795
2: -6.5584602, -4.9863763, -6.6177559, -4.9731340, -0.8204173, 0.8230046
3: -11.4280176, -9.4958572, -11.4850903, -9.4167938, -0.9535089, 0.9375939
4: -4.3297524, -2.7941709, -4.3751445, -2.7157457, -0.9663650, 0.9691899
5: -12.3438921, -10.5488529, -12.3498554, -10.5275106, -0.7956005, 0.7990105
6: -10.0624084, -8.0437260, -10.0905256, -8.0385056, -0.9513292, 0.9935408
7: -4.2180338, -2.6888964, -4.2485495, -2.6703827, -0.6961493, 0.6964760
8: -3.3258572, -1.8531594, -3.3621249, -1.8348875, -0.5790137, 0.5768494
9: -12.0421772, -10.4722948, -12.0791082, -10.4300518, -0.7298208, 0.7095125

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 6220

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 820

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944291, upper bound: 0.2988515
time: 6.28 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2944291, upper bound: 0.2967329
time: 4.31 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9577265, -7.0447598, -8.9330759, -7.0464821, -1.0718479, 1.0549915
1: 2.4160821, 3.9457879, 2.4182973, 3.9110363, -0.7903020, 0.8173904
2: -6.5345521, -4.9895291, -6.5156131, -4.9922833, -0.8086776, 0.7985013
3: -11.4782810, -9.5055122, -11.4204941, -9.5074329, -0.9601924, 0.9073792
4: -4.3584814, -2.7489815, -4.3536558, -2.8071930, -0.9556206, 0.9890120
5: -12.3429766, -10.5807953, -12.3370571, -10.5825748, -0.7744077, 0.7693608
6: -10.0652943, -8.0694914, -10.0601902, -8.0909300, -0.9340944, 0.9527783
7: -4.2385731, -2.6925666, -4.2126069, -2.6939704, -0.7038105, 0.6833698
8: -3.3136072, -1.8450727, -3.2898531, -1.8457222, -0.5834438, 0.5580812
9: -12.0500517, -10.4460583, -12.0047302, -10.4481449, -0.7412424, 0.7075948

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2966790, upper bound: 0.2941710
time: 4.51 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2966815, upper bound: 0.2941921
time: 4.54 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9829321, -7.0262623, -8.9330759, -7.0464821, -1.0868635, 1.0605228
1: 2.4004331, 3.9858456, 2.4182973, 3.9110363, -0.8074090, 0.8243058
2: -6.6077971, -4.9752917, -6.5156131, -4.9922833, -0.8229858, 0.8046666
3: -11.4853516, -9.4330921, -11.4204941, -9.5074329, -0.9628634, 0.9413886
4: -4.3706927, -2.7183371, -4.3536558, -2.8071930, -0.9649762, 0.9923420
5: -12.3499451, -10.5344591, -12.3370571, -10.5825748, -0.7777902, 0.7874391
6: -10.0816565, -8.0384140, -10.0601902, -8.0909300, -0.9502020, 0.9586415
7: -4.2488689, -2.6765530, -4.2126069, -2.6939704, -0.7099204, 0.6932113
8: -3.3540797, -1.8361249, -3.2898531, -1.8457222, -0.5909139, 0.5673099
9: -12.0792456, -10.4348316, -12.0047302, -10.4481449, -0.7446620, 0.7190825

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5734
type: B, layer: 1, pos: 6220
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 820

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5734

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.8209632635116577
rel_dist={1: [-0.29885889706970215, 0.2988555407171436]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1653.32 seconds
