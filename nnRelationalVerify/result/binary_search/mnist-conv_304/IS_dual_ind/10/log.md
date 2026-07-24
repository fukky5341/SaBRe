## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0070000638
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.1958132, 3.1958132)
1: (-10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.7144113, 2.7144113)
2: (-10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.8229380, 2.8229380)
3: (-12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.4258022, 2.4258022)
4: (5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.5450792, 2.5450792)
5: (-8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.6159039, 2.6159039)
6: (-12.7108383, -9.7072067, -12.7108383, -9.7072067, -3.0036316, 3.0036316)
7: (-6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743)
8: (-3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.7740321, 2.7740321)
9: (-5.4689426, -3.2161665, -5.4689426, -3.2161665, -2.2485318, 2.2485321)

## BASE Result
execution time: IAR + LP analysis = 14.56 + 34.67 = 49.22 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3550.78 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.4249300956726074
rel_dist={4: [-1.3292853219987384, 1.3292848273935602]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.132899761199951
rel_dist={4: [-0.7676964433562325, 0.7676953753322513]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.191305637359619
rel_dist={4: [-0.8893789777736325, 0.8893816357061688]}

## Binary Search Result
Binary search time: 208.45 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3342.32 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053862, upper bound: 1.4213007
time: 7.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212972, upper bound: 1.4212981
time: 5.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.97 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.97
Output dim: 4, lower bound: -1.4053862, upper bound: 1.4213007
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.97
Output dim: 4, lower bound: -1.4212972, upper bound: 1.4212981

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2714777, -11.1257515, -14.2722282, -11.0764151, -2.9955444, 2.9474893
1: -10.6126785, -7.9058170, -10.6166239, -7.9022126, -2.3846941, 2.3849111
2: -10.1411486, -7.3414025, -10.1443129, -7.3213749, -2.6913567, 2.6765318
3: -12.7772465, -10.3578882, -12.7821178, -10.3563156, -2.2369590, 2.2403417
4: 5.9066377, 8.4296684, 5.8858538, 8.4309330, -2.4620528, 2.4820247
5: -8.3625040, -5.7561874, -8.3676176, -5.7517138, -2.2860923, 2.2876155
6: -12.7085762, -9.7807646, -12.7108383, -9.7072067, -2.6600199, 2.5883126
7: -6.2112141, -3.3355632, -6.2174892, -3.3342149, -2.8769991, 2.8819261
8: -2.9995489, -0.2421093, -3.0022964, -0.2282643, -2.5045290, 2.4929280
9: -5.4626245, -3.2182441, -5.4689426, -3.2161665, -1.9221783, 1.9262805

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053858, upper bound: 1.4053860
time: 6.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053858, upper bound: 1.4212984
time: 7.13 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.4242506, -11.0621929, -14.2722273, -11.0764236, -3.0390377, 3.0349236
1: -10.6239071, -7.8924012, -10.6166229, -7.9022150, -2.3961921, 2.3982766
2: -10.2043324, -7.3152981, -10.1443119, -7.3213873, -2.7438278, 2.7164977
3: -12.7920551, -10.3471174, -12.7821150, -10.3563166, -2.2525740, 2.2505722
4: 5.8710923, 8.4850378, 5.8858614, 8.4309301, -2.5011711, 2.5046725
5: -8.3798885, -5.7470326, -8.3676167, -5.7517166, -2.3090658, 2.2960525
6: -12.9543877, -9.7024460, -12.7108402, -9.7072315, -2.6990404, 2.6839771
7: -6.2408209, -3.3283496, -6.2174864, -3.3342156, -2.9066052, 2.8891368
8: -3.0490851, -0.2229128, -3.0022945, -0.2282724, -2.5542502, 2.5141029
9: -5.4763236, -3.1923084, -5.4689379, -3.2161665, -1.9352870, 1.9529910

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941514, upper bound: 1.4186331
time: 4.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212802, upper bound: 1.4212824
time: 7.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.97
Output dim: 4, lower bound: -1.4053858, upper bound: 1.4053860
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.97
Output dim: 4, lower bound: -1.4053858, upper bound: 1.4212984
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.97
Output dim: 4, lower bound: -1.3941514, upper bound: 1.4186331
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.97
Output dim: 4, lower bound: -1.4212802, upper bound: 1.4212824

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.2714777, -11.1257515, -14.2714777, -11.1257515, -2.9430785, 2.9430785
1: -10.6126785, -7.9058170, -10.6126785, -7.9058170, -2.3808866, 2.3808863
2: -10.1411486, -7.3414025, -10.1411486, -7.3414025, -2.6683969, 2.6683965
3: -12.7772465, -10.3578882, -12.7772465, -10.3578882, -2.2355127, 2.2355127
4: 5.9066377, 8.4296684, 5.9066377, 8.4296684, -2.4607420, 2.4607420
5: -8.3625040, -5.7561874, -8.3625040, -5.7561874, -2.2804041, 2.2804041
6: -12.7085762, -9.7807646, -12.7085762, -9.7807646, -2.5864801, 2.5864797
7: -6.2112141, -3.3355632, -6.2112141, -3.3355632, -2.8756509, 2.8756509
8: -2.9995489, -0.2421093, -2.9995489, -0.2421093, -2.4906406, 2.4906402
9: -5.4626245, -3.2182441, -5.4626245, -3.2182441, -1.9198680, 1.9198678

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4028997, upper bound: 1.3784122
time: 4.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053700, upper bound: 1.4053684
time: 6.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.2714777, -11.1257515, -14.4242506, -11.0621929, -3.0053611, 2.9865227
1: -10.6126785, -7.9058170, -10.6239071, -7.8924012, -2.3942542, 2.3918500
2: -10.1411486, -7.3414025, -10.2043324, -7.3152981, -2.6946950, 2.7209051
3: -12.7772465, -10.3578882, -12.7920551, -10.3471174, -2.2457457, 2.2507930
4: 5.9066377, 8.4296684, 5.8710923, 8.4850378, -2.4833679, 2.4971647
5: -8.3625040, -5.7561874, -8.3798885, -5.7470326, -2.2888446, 2.2981527
6: -12.7085762, -9.7807646, -12.9543877, -9.7024460, -2.6648674, 2.6254370
7: -6.2112141, -3.3355632, -6.2408209, -3.3283496, -2.8828645, 2.9052577
8: -2.9995489, -0.2421093, -3.0490851, -0.2229128, -2.5097914, 2.5403690
9: -5.4626245, -3.2182441, -5.4763236, -3.1923084, -1.9465568, 1.9317150

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4028997, upper bound: 1.3941524
time: 4.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053700, upper bound: 1.4212819
time: 4.67 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.4163685, -11.0662766, -14.2392712, -11.0884399, -3.0190105, 3.0002689
1: -10.6183729, -7.9009700, -10.5863590, -7.9344072, -2.3582249, 2.3319921
2: -10.2013340, -7.3261132, -10.1246023, -7.3621464, -2.6931291, 2.6764102
3: -12.7840652, -10.3507957, -12.7497883, -10.3731213, -2.2208214, 2.2138076
4: 5.8775396, 8.4701757, 5.9232922, 8.3751202, -2.4332914, 2.3851964
5: -8.3743057, -5.7548881, -8.3391533, -5.7830033, -2.2719646, 2.2557817
6: -12.9518194, -9.7124844, -12.6938305, -9.7448235, -2.6561852, 2.5826674
7: -6.2238979, -3.3323622, -6.1539965, -3.3660161, -2.8364553, 2.8216343
8: -3.0454984, -0.2277889, -2.9860802, -0.2476435, -2.5264311, 2.4794335
9: -5.4562893, -3.1967702, -5.3939877, -3.2494373, -1.7571011, 1.8752062

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941514, upper bound: 1.3941521
time: 4.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941514, upper bound: 1.4186348
time: 4.27 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.4242506, -11.0621929, -14.2722225, -11.0764236, -3.0390363, 3.0632021
1: -10.6239071, -7.8924012, -10.6166191, -7.9022212, -2.3961821, 2.4134698
2: -10.2043324, -7.3152981, -10.1443100, -7.3213964, -2.7438254, 2.7191062
3: -12.7920551, -10.3471174, -12.7821102, -10.3563194, -2.2647462, 2.2505651
4: 5.8710923, 8.4850378, 5.8858671, 8.4309254, -2.4838104, 2.4906425
5: -8.3798885, -5.7470326, -8.3676128, -5.7517242, -2.3090591, 2.3016150
6: -12.9543877, -9.7024460, -12.7108364, -9.7072363, -2.6943769, 2.6839750
7: -6.2408209, -3.3283496, -6.2174749, -3.3342195, -2.9066014, 2.8891253
8: -3.0490851, -0.2229128, -3.0022931, -0.2282763, -2.5532355, 2.5140996
9: -5.4763236, -3.1923084, -5.4689212, -3.2161689, -1.9347789, 1.9318585

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4186326, upper bound: 1.3941549
time: 4.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4186324, upper bound: 1.3941526
time: 7.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.97 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.4028997, upper bound: 1.3784122
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.4053700, upper bound: 1.4053684
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.4028997, upper bound: 1.3941524
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.4053700, upper bound: 1.4212819
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.3941514, upper bound: 1.3941521
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.3941514, upper bound: 1.4186348
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.4186326, upper bound: 1.3941549
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.97
Output dim: 4, lower bound: -1.4186324, upper bound: 1.3941526

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.2636127, -11.1298332, -2.9083991, 2.9248552
1: -10.5824461, -7.9380188, -10.6071577, -7.9143944, -2.3146224, 2.3429782
2: -10.1214695, -7.3821011, -10.1381645, -7.3521652, -2.6281776, 2.6190810
3: -12.7449169, -10.3747272, -12.7692471, -10.3615894, -2.1987262, 2.2033370
4: 5.9440694, 8.3738632, 5.9130220, 8.4148016, -2.3544035, 2.3929062
5: -8.3340082, -5.7874660, -8.3568916, -5.7640505, -2.2400494, 2.2432740
6: -12.6915712, -9.8183794, -12.7060099, -9.7908154, -2.4852157, 2.5437913
7: -6.1476965, -3.3673813, -6.1942725, -3.3395875, -2.8081090, 2.8049340
8: -2.9833360, -0.2614660, -2.9959626, -0.2469645, -2.4559889, 2.4628315
9: -5.3876762, -3.2515135, -5.4425669, -3.2227125, -1.8421168, 1.7416842

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784132, upper bound: 1.3784129
time: 5.68 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784132, upper bound: 1.3784155
time: 5.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.2714777, -11.1257515, -2.9713516, 2.9410129
1: -10.6126757, -7.9058218, -10.6126785, -7.9058170, -2.3960772, 2.3808761
2: -10.1411448, -7.3414111, -10.1411486, -7.3414025, -2.6710072, 2.6683884
3: -12.7772408, -10.3578930, -12.7772465, -10.3578882, -2.2355056, 2.2476859
4: 5.9066410, 8.4296608, 5.9066377, 8.4296684, -2.4607382, 2.4433804
5: -8.3625011, -5.7561960, -8.3625040, -5.7561874, -2.2859669, 2.2803974
6: -12.7085724, -9.7807722, -12.7085762, -9.7807646, -2.5864782, 2.5818164
7: -6.2112017, -3.3355670, -6.2112141, -3.3355632, -2.8756385, 2.8756471
8: -2.9995470, -0.2421117, -2.9995489, -0.2421093, -2.4906368, 2.4896240
9: -5.4626083, -3.2182460, -5.4626245, -3.2182441, -1.8987460, 1.9194717

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784132, upper bound: 1.4028997
time: 5.88 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784134, upper bound: 1.4029022
time: 5.15 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.4163685, -11.0662766, -2.9706945, 2.9664843
1: -10.5824461, -7.9380188, -10.6183729, -7.9009700, -2.3280120, 2.3539155
2: -10.1214695, -7.3821011, -10.2013340, -7.3261132, -2.6545839, 2.6701865
3: -12.7449169, -10.3747272, -12.7840652, -10.3507957, -2.2089758, 2.2190232
4: 5.9440694, 8.3738632, 5.8775396, 8.4701757, -2.3640051, 2.4292789
5: -8.3340082, -5.7874660, -8.3743057, -5.7548881, -2.2484984, 2.2610612
6: -12.6915712, -9.8183794, -12.9518194, -9.7124844, -2.5635972, 2.5825636
7: -6.1476965, -3.3673813, -6.2238979, -3.3323622, -2.8153343, 2.8338356
8: -2.9833360, -0.2614660, -3.0454984, -0.2277889, -2.4751682, 2.5125551
9: -5.3876762, -3.2515135, -5.4562893, -3.1967702, -1.8687732, 1.7534997

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784128, upper bound: 1.3941512
time: 8.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784128, upper bound: 1.3941536
time: 7.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.4242506, -11.0621929, -3.0336404, 2.9865210
1: -10.6126757, -7.9058218, -10.6239071, -7.8924012, -2.4094481, 2.3918400
2: -10.1411448, -7.3414111, -10.2043324, -7.3152981, -2.6973057, 2.7209024
3: -12.7772408, -10.3578930, -12.7920551, -10.3471174, -2.2457390, 2.2629664
4: 5.9066410, 8.4296608, 5.8710923, 8.4850378, -2.4693146, 2.4798036
5: -8.3625011, -5.7561960, -8.3798885, -5.7470326, -2.2944074, 2.2981458
6: -12.7085724, -9.7807722, -12.9543877, -9.7024460, -2.6648655, 2.6207738
7: -6.2112017, -3.3355670, -6.2408209, -3.3283496, -2.8828521, 2.9052539
8: -2.9995470, -0.2421117, -3.0490851, -0.2229128, -2.5097885, 2.5393529
9: -5.4626083, -3.2182460, -5.4763236, -3.1923084, -1.9254239, 1.9311936

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784128, upper bound: 1.4186322
time: 7.76 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3784130, upper bound: 1.4212813
time: 7.65 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.2392712, -11.0884399, -2.9910908, 2.9896095
1: -10.5936718, -7.9245777, -10.5863590, -7.9344072, -2.3050072, 2.3071728
2: -10.1846275, -7.3561811, -10.1246023, -7.3621464, -2.6678963, 2.6413445
3: -12.7597809, -10.3639021, -12.7497883, -10.3731213, -2.1958027, 2.1934156
4: 5.9087772, 8.4292564, 5.9232922, 8.3751202, -2.3497109, 2.3557954
5: -8.3513088, -5.7782631, -8.3391533, -5.7830033, -2.2461538, 2.2330999
6: -12.9373684, -9.7400455, -12.6938305, -9.7448235, -2.5777588, 2.5598643
7: -6.1773005, -3.3601122, -6.1539965, -3.3660161, -2.7898035, 2.7724099
8: -3.0328588, -0.2423491, -2.9860802, -0.2476435, -2.5017405, 2.4616480
9: -5.4014525, -3.2255373, -5.3939877, -3.2494373, -1.7078948, 1.7254984

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941514, upper bound: 1.3784189
time: 4.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941527, upper bound: 1.3795774
time: 5.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4242449, -11.0621948, -14.2392712, -11.0884399, -3.0179367, 3.0005698
1: -10.6239023, -7.8924074, -10.5863590, -7.9344072, -2.3595867, 2.3410659
2: -10.2043276, -7.3153048, -10.1246023, -7.3621464, -2.6891789, 2.6892986
3: -12.7920494, -10.3471184, -12.7497883, -10.3731213, -2.2289343, 2.2139335
4: 5.8710976, 8.4850273, 5.9232922, 8.3751202, -2.4380016, 2.3895640
5: -8.3798857, -5.7470388, -8.3391533, -5.7830033, -2.2764912, 2.2644508
6: -12.9543858, -9.7024517, -12.6938305, -9.7448235, -2.6529698, 2.5997405
7: -6.2408094, -3.3283525, -6.1539965, -3.3660161, -2.8469892, 2.8256440
8: -3.0490832, -0.2229147, -2.9860802, -0.2476435, -2.5301371, 2.4858532
9: -5.4763069, -3.1923113, -5.3939877, -3.2494373, -1.7593861, 1.8601696

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941514, upper bound: 1.4028987
time: 4.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941526, upper bound: 1.4040548
time: 6.10 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.2722225, -11.0764236, -3.0007186, 3.0160394
1: -10.5936718, -7.9245777, -10.6166191, -7.9022212, -2.3389280, 2.3618243
2: -10.1846275, -7.3561811, -10.1443100, -7.3213964, -2.6960263, 2.6700034
3: -12.7597809, -10.3639021, -12.7821102, -10.3563194, -2.2159896, 2.2266159
4: 5.9087772, 8.4292564, 5.8858671, 8.4309254, -2.4002619, 2.4283922
5: -8.3513088, -5.7782631, -8.3676128, -5.7517242, -2.2706437, 2.2635469
6: -12.9373684, -9.7400455, -12.7108364, -9.7072363, -2.6031408, 2.6434951
7: -6.1773005, -3.3601122, -6.2174749, -3.3342195, -2.8430810, 2.8298407
8: -3.0328588, -0.2423491, -3.0022931, -0.2282763, -2.5259428, 2.4899440
9: -5.4014525, -3.2255373, -5.4689212, -3.2161689, -1.8601954, 1.7593093

Time for backsubstitution: 15.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941514, upper bound: 1.3784118
time: 4.99 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941527, upper bound: 1.3795710
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4242449, -11.0621948, -14.2722225, -11.0764236, -3.0625300, 3.0631962
1: -10.6239023, -7.8924074, -10.6166191, -7.9022212, -2.4113731, 2.4134622
2: -10.2043276, -7.3153048, -10.1443100, -7.3213964, -2.7389207, 2.7190986
3: -12.7920494, -10.3471184, -12.7821102, -10.3563194, -2.2647405, 2.2627380
4: 5.8710976, 8.4850273, 5.8858671, 8.4309254, -2.4838076, 2.4786768
5: -8.3798857, -5.7470388, -8.3676128, -5.7517242, -2.3140795, 2.3016090
6: -12.9543858, -9.7024517, -12.7108364, -9.7072363, -2.6904116, 2.6793108
7: -6.2408094, -3.3283525, -6.2174749, -3.3342195, -2.9065900, 2.8891225
8: -3.0490832, -0.2229147, -3.0022931, -0.2282763, -2.5532317, 2.5130825
9: -5.4763069, -3.1923113, -5.4689212, -3.2161689, -1.9141626, 1.9202478

Time for backsubstitution: 15.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941516, upper bound: 1.4053693
time: 8.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941528, upper bound: 1.4066159
time: 10.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784132, upper bound: 1.3784129
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784132, upper bound: 1.3784155
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784132, upper bound: 1.4028997
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784134, upper bound: 1.4029022
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784128, upper bound: 1.3941512
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784128, upper bound: 1.3941536
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784128, upper bound: 1.4186322
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3784130, upper bound: 1.4212813
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941514, upper bound: 1.3784189
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941527, upper bound: 1.3795774
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941514, upper bound: 1.4028987
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941526, upper bound: 1.4040548
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941514, upper bound: 1.3784118
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941527, upper bound: 1.3795710
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941516, upper bound: 1.4053693
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.99
Output dim: 4, lower bound: -1.3941528, upper bound: 1.4066159

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.2385406, -11.1378107, -2.8977284, 2.8977284
1: -10.5824461, -7.9380188, -10.5824461, -7.9380188, -2.2897835, 2.2897832
2: -10.1214695, -7.3821011, -10.1214695, -7.3821011, -2.5930963, 2.5930963
3: -12.7449169, -10.3747272, -12.7449169, -10.3747272, -2.1782556, 2.1782558
4: 5.9440694, 8.3738632, 5.9440694, 8.3738632, -2.3105226, 2.3105216
5: -8.3340082, -5.7874660, -8.3340082, -5.7874660, -2.2173166, 2.2173166
6: -12.6915712, -9.8183794, -12.6915712, -9.8183794, -2.4624877, 2.4624877
7: -6.1476965, -3.3673813, -6.1476965, -3.3673813, -2.7582817, 2.7582817
8: -2.9833360, -0.2614660, -2.9833360, -0.2614660, -2.4382210, 2.4382205
9: -5.3876762, -3.2515135, -5.3876762, -3.2515135, -1.6923513, 1.6923511

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783959, upper bound: 1.3693584
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783958, upper bound: 1.3783926
time: 6.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.2714739, -11.1257534, -2.9086905, 2.9241865
1: -10.5824461, -7.9380188, -10.6126757, -7.9058218, -2.3237081, 2.3443143
2: -10.1214695, -7.3821011, -10.1411448, -7.3414111, -2.6411047, 2.6217594
3: -12.7449169, -10.3747272, -12.7772408, -10.3578930, -2.1988688, 2.2114596
4: 5.9440694, 8.3738632, 5.9066410, 8.4296608, -2.3615103, 2.3985009
5: -8.3340082, -5.7874660, -8.3625011, -5.7561960, -2.2487268, 2.2478385
6: -12.6915712, -9.8183794, -12.7085724, -9.7807722, -2.5023232, 2.5460336
7: -6.1476965, -3.3673813, -6.2112017, -3.3355670, -2.8121295, 2.8157997
8: -2.9833360, -0.2614660, -2.9995470, -0.2421117, -2.4624138, 2.4665332
9: -5.3876762, -3.2515135, -5.4626083, -3.2182460, -1.8448305, 1.7439611

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783957, upper bound: 1.3693595
time: 5.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783957, upper bound: 1.3783926
time: 6.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.2385406, -11.1378107, -2.9241862, 2.9086914
1: -10.6126757, -7.9058218, -10.5824461, -7.9380188, -2.3443141, 2.3237083
2: -10.1411448, -7.3414111, -10.1214695, -7.3821011, -2.6217594, 2.6411049
3: -12.7772408, -10.3578930, -12.7449169, -10.3747272, -2.2114596, 2.1988688
4: 5.9066410, 8.4296608, 5.9440694, 8.3738632, -2.3985004, 2.3615103
5: -8.3625011, -5.7561960, -8.3340082, -5.7874660, -2.2478385, 2.2487268
6: -12.7085724, -9.7807722, -12.6915712, -9.8183794, -2.5460339, 2.5023234
7: -6.2112017, -3.3355670, -6.1476965, -3.3673813, -2.8157997, 2.8121295
8: -2.9995470, -0.2421117, -2.9833360, -0.2614660, -2.4665327, 2.4624138
9: -5.4626083, -3.2182460, -5.3876762, -3.2515135, -1.7439613, 1.8448303

Time for backsubstitution: 15.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783908, upper bound: 1.3938443
time: 5.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783906, upper bound: 1.4028781
time: 6.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.2714739, -11.1257534, -2.9713449, 2.9713449
1: -10.6126757, -7.9058218, -10.6126757, -7.9058218, -2.3960662, 2.3960662
2: -10.1411448, -7.3414111, -10.1411448, -7.3414111, -2.6709995, 2.6709991
3: -12.7772408, -10.3578930, -12.7772408, -10.3578930, -2.2476802, 2.2476797
4: 5.9066410, 8.4296608, 5.9066410, 8.4296608, -2.4433780, 2.4433775
5: -8.3625011, -5.7561960, -8.3625011, -5.7561960, -2.2859607, 2.2859607
6: -12.7085724, -9.7807722, -12.7085724, -9.7807722, -2.5818148, 2.5818145
7: -6.2112017, -3.3355670, -6.2112017, -3.3355670, -2.8756347, 2.8756347
8: -2.9995470, -0.2421117, -2.9995470, -0.2421117, -2.4896212, 2.4896212
9: -5.4626083, -3.2182460, -5.4626083, -3.2182460, -1.8987446, 1.8987443

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783907, upper bound: 1.3938483
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783908, upper bound: 1.4053498
time: 5.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.3912497, -11.0742731, -2.9600306, 2.9385648
1: -10.5824461, -7.9380188, -10.5936718, -7.9245777, -2.3031929, 2.3006864
2: -10.1214695, -7.3821011, -10.1846275, -7.3561811, -2.6193709, 2.6449499
3: -12.7449169, -10.3747272, -12.7597809, -10.3639021, -2.1885643, 2.1940050
4: 5.9440694, 8.3738632, 5.9087772, 8.4292564, -2.3346038, 2.3457131
5: -8.3340082, -5.7874660, -8.3513088, -5.7782631, -2.2258167, 2.2352500
6: -12.6915712, -9.8183794, -12.9373684, -9.7400455, -2.5407948, 2.5041475
7: -6.1476965, -3.3673813, -6.1773005, -3.3601122, -2.7657309, 2.7871847
8: -2.9833360, -0.2614660, -3.0328588, -0.2423491, -2.4573832, 2.4878697
9: -5.3876762, -3.2515135, -5.4014525, -3.2255373, -1.7190685, 1.7043066

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783954, upper bound: 1.3851188
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783954, upper bound: 1.3941280
time: 6.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.4242449, -11.0621948, -2.9709797, 2.9654102
1: -10.5824461, -7.9380188, -10.6239023, -7.8924074, -2.3370857, 2.3552773
2: -10.1214695, -7.3821011, -10.2043276, -7.3153048, -2.6675177, 2.6662364
3: -12.7449169, -10.3747272, -12.7920494, -10.3471184, -2.2091017, 2.2271366
4: 5.9440694, 8.3738632, 5.8710976, 8.4850273, -2.3683724, 2.4349232
5: -8.3340082, -5.7874660, -8.3798857, -5.7470388, -2.2571678, 2.2655878
6: -12.6915712, -9.8183794, -12.9543858, -9.7024517, -2.5806699, 2.5793481
7: -6.1476965, -3.3673813, -6.2408094, -3.3283525, -2.8193440, 2.8443842
8: -2.9833360, -0.2614660, -3.0490832, -0.2229147, -2.4815884, 2.5162611
9: -5.3876762, -3.2515135, -5.4763069, -3.1923113, -1.8537362, 1.7557847

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783955, upper bound: 1.3851188
time: 4.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783953, upper bound: 1.3941308
time: 5.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.3912497, -11.0742731, -2.9864883, 2.9481883
1: -10.6126757, -7.9058218, -10.5936718, -7.9245777, -2.3578019, 2.3346114
2: -10.1411448, -7.3414111, -10.1846275, -7.3561811, -2.6480060, 2.6730576
3: -12.7772408, -10.3578930, -12.7597809, -10.3639021, -2.2217679, 2.2142091
4: 5.9066410, 8.4296608, 5.9087772, 8.4292564, -2.4070635, 2.3963327
5: -8.3625011, -5.7561960, -8.3513088, -5.7782631, -2.2563386, 2.2634902
6: -12.7085724, -9.7807722, -12.9373684, -9.7400455, -2.6243868, 2.5295436
7: -6.2112017, -3.3355670, -6.1773005, -3.3601122, -2.8231244, 2.8417335
8: -2.9995470, -0.2421117, -3.0328588, -0.2423491, -2.4856329, 2.5120625
9: -5.4626083, -3.2182460, -5.4014525, -3.2255373, -1.7528825, 1.8566096

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783904, upper bound: 1.4095904
time: 5.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783904, upper bound: 1.4186079
time: 6.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.4242449, -11.0621948, -3.0336342, 3.0100148
1: -10.6126757, -7.9058218, -10.6239023, -7.8924074, -2.4094400, 2.4070299
2: -10.1411448, -7.3414111, -10.2043276, -7.3153048, -2.6972980, 2.7159986
3: -12.7772408, -10.3578930, -12.7920494, -10.3471184, -2.2579122, 2.2629604
4: 5.9066410, 8.4296608, 5.8710976, 8.4850273, -2.4573488, 2.4797997
5: -8.3625011, -5.7561960, -8.3798857, -5.7470388, -2.2944016, 2.3037100
6: -12.7085724, -9.7807722, -12.9543858, -9.7024517, -2.6602030, 2.6168089
7: -6.2112017, -3.3355670, -6.2408094, -3.3283525, -2.8828492, 2.9052424
8: -2.9995470, -0.2421117, -3.0490832, -0.2229147, -2.5087729, 2.5393496
9: -5.4626083, -3.2182460, -5.4763069, -3.1923113, -1.9138129, 1.9105909

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783902, upper bound: 1.4121027
time: 4.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783904, upper bound: 1.4212608
time: 5.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.2385406, -11.1378107, -2.9385653, 2.9600306
1: -10.5936718, -7.9245777, -10.5824461, -7.9380188, -2.3006864, 2.3031926
2: -10.1846275, -7.3561811, -10.1214695, -7.3821011, -2.6449499, 2.6193712
3: -12.7597809, -10.3639021, -12.7449169, -10.3747272, -2.1940050, 2.1885641
4: 5.9087772, 8.4292564, 5.9440694, 8.3738632, -2.3457131, 2.3346038
5: -8.3513088, -5.7782631, -8.3340082, -5.7874660, -2.2352500, 2.2258167
6: -12.9373684, -9.7400455, -12.6915712, -9.8183794, -2.5041475, 2.5407948
7: -6.1773005, -3.3601122, -6.1476965, -3.3673813, -2.7871847, 2.7657313
8: -3.0328588, -0.2423491, -2.9833360, -0.2614660, -2.4878693, 2.4573827
9: -5.4014525, -3.2255373, -5.3876762, -3.2515135, -1.7043066, 1.7190683

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941363, upper bound: 1.3693631
time: 7.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941341, upper bound: 1.3783960
time: 4.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.3912497, -11.0742731, -2.9935608, 2.9935608
1: -10.5936718, -7.9245777, -10.5936718, -7.9245777, -2.3147571, 2.3147569
2: -10.1846275, -7.3561811, -10.1846275, -7.3561811, -2.6522107, 2.6522107
3: -12.7597809, -10.3639021, -12.7597809, -10.3639021, -2.2002387, 2.2002385
4: 5.9087772, 8.4292564, 5.9087772, 8.4292564, -2.3694263, 2.3694263
5: -8.3513088, -5.7782631, -8.3513088, -5.7782631, -2.2501335, 2.2501333
6: -12.9373684, -9.7400455, -12.9373684, -9.7400455, -2.5761080, 2.5761077
7: -6.1773005, -3.3601122, -6.1773005, -3.3601122, -2.7957492, 2.7957497
8: -3.0328588, -0.2423491, -3.0328588, -0.2423491, -2.4823322, 2.4823322
9: -5.4014525, -3.2255373, -5.4014525, -3.2255373, -1.7163033, 1.7163033

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941375, upper bound: 1.3705386
time: 6.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941353, upper bound: 1.3795551
time: 4.29 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4242449, -11.0621948, -14.2385406, -11.1378107, -2.9654102, 2.9709792
1: -10.6239023, -7.8924074, -10.5824461, -7.9380188, -2.3552771, 2.3370857
2: -10.2043276, -7.3153048, -10.1214695, -7.3821011, -2.6662364, 2.6675184
3: -12.7920494, -10.3471184, -12.7449169, -10.3747272, -2.2271366, 2.2091017
4: 5.8710976, 8.4850273, 5.9440694, 8.3738632, -2.4349232, 2.3683724
5: -8.3798857, -5.7470388, -8.3340082, -5.7874660, -2.2655878, 2.2571676
6: -12.9543858, -9.7024517, -12.6915712, -9.8183794, -2.5793481, 2.5806699
7: -6.2408094, -3.3283525, -6.1476965, -3.3673813, -2.8443842, 2.8193440
8: -3.0490832, -0.2229147, -2.9833360, -0.2614660, -2.5162611, 2.4815884
9: -5.4763069, -3.1923113, -5.3876762, -3.2515135, -1.7557845, 1.8537366

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941288, upper bound: 1.3938435
time: 9.13 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941288, upper bound: 1.4028746
time: 4.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4242449, -11.0621948, -14.3912497, -11.0742731, -3.0200243, 3.0045223
1: -10.6239023, -7.8924074, -10.5936718, -7.9245777, -2.3694267, 2.3486497
2: -10.2043276, -7.3153048, -10.1846275, -7.3561811, -2.6807022, 2.7001648
3: -12.7920494, -10.3471184, -12.7597809, -10.3639021, -2.2333703, 2.2203665
4: 5.8710976, 8.4850273, 5.9087772, 8.4292564, -2.4432573, 2.4031949
5: -8.3798857, -5.7470388, -8.3513088, -5.7782631, -2.2804708, 2.2743025
6: -12.9543858, -9.7024517, -12.9373684, -9.7400455, -2.6578703, 2.6079500
7: -6.2408094, -3.3283525, -6.1773005, -3.3601122, -2.8526049, 2.8489480
8: -3.0490832, -0.2229147, -3.0328588, -0.2423491, -2.5106616, 2.5065379
9: -5.4763069, -3.1923113, -5.4014525, -3.2255373, -1.7647061, 1.8655162

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941301, upper bound: 1.3950035
time: 7.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941301, upper bound: 1.4040279
time: 4.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.2714739, -11.1257534, -2.9481883, 2.9864888
1: -10.5936718, -7.9245777, -10.6126757, -7.9058218, -2.3346114, 2.3578022
2: -10.1846275, -7.3561811, -10.1411448, -7.3414111, -2.6730576, 2.6480060
3: -12.7597809, -10.3639021, -12.7772408, -10.3578930, -2.2142086, 2.2217679
4: 5.9087772, 8.4292564, 5.9066410, 8.4296608, -2.3963327, 2.4070640
5: -8.3513088, -5.7782631, -8.3625011, -5.7561960, -2.2634902, 2.2563386
6: -12.9373684, -9.7400455, -12.7085724, -9.7807722, -2.5295434, 2.6243868
7: -6.1773005, -3.3601122, -6.2112017, -3.3355670, -2.8417335, 2.8231244
8: -3.0328588, -0.2423491, -2.9995470, -0.2421117, -2.5120630, 2.4856329
9: -5.4014525, -3.2255373, -5.4626083, -3.2182460, -1.8566093, 1.7528825

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4186100, upper bound: 1.3693605
time: 5.22 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4186108, upper bound: 1.3783919
time: 6.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.4242449, -11.0621948, -3.0045223, 3.0200253
1: -10.5936718, -7.9245777, -10.6239023, -7.8924074, -2.3486500, 2.3694267
2: -10.1846275, -7.3561811, -10.2043276, -7.3153048, -2.7001648, 2.6807027
3: -12.7597809, -10.3639021, -12.7920494, -10.3471184, -2.2203665, 2.2333698
4: 5.9087772, 8.4292564, 5.8710976, 8.4850273, -2.4031949, 2.4432569
5: -8.3513088, -5.7782631, -8.3798857, -5.7470388, -2.2743025, 2.2804711
6: -12.9373684, -9.7400455, -12.9543858, -9.7024517, -2.6079497, 2.6578703
7: -6.1773005, -3.3601122, -6.2408094, -3.3283525, -2.8489480, 2.8526046
8: -3.0328588, -0.2423491, -3.0490832, -0.2229147, -2.5065379, 2.5106616
9: -5.4014525, -3.2255373, -5.4763069, -3.1923113, -1.8655159, 1.7647061

Time for backsubstitution: 14.80 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.483335494995117
rel_dist={4: [-1.421316393236534, 1.4213159903507133]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1147588, upper bound: 1.1255431
time: 14.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255876, upper bound: 1.1255885
time: 6.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.49 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.49
Output dim: 4, lower bound: -1.1147588, upper bound: 1.1255431
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.49
Output dim: 4, lower bound: -1.1255876, upper bound: 1.1255885

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2714777, -11.1257515, -14.2720470, -11.0880442, -2.5856853, 2.5489488
1: -10.6126785, -7.9058170, -10.6156988, -7.9030633, -2.1122494, 2.1124165
2: -10.1411486, -7.3414025, -10.1435709, -7.3261056, -2.4075384, 2.3962059
3: -12.7772465, -10.3578882, -12.7809677, -10.3566971, -2.0141859, 2.0167756
4: 5.9066377, 8.4296684, 5.8907566, 8.4306364, -2.2865267, 2.3017826
5: -8.3625040, -5.7561874, -8.3664169, -5.7527733, -2.0353589, 2.0365338
6: -12.7085762, -9.7807646, -12.7103062, -9.7245388, -2.3066859, 2.2518747
7: -6.2112141, -3.3355632, -6.2159996, -3.3345418, -2.7864790, 2.7897253
8: -2.9995489, -0.2421093, -3.0016437, -0.2315273, -2.2918377, 2.2829690
9: -5.4626245, -3.2182441, -5.4674530, -3.2166538, -1.7296705, 1.7328026

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003205, upper bound: 1.1242446
time: 5.17 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1147428, upper bound: 1.1255307
time: 5.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.4242506, -11.0621929, -14.2722263, -11.0764284, -2.6384234, 2.6116512
1: -10.6239071, -7.8924012, -10.6166220, -7.9022155, -2.1245942, 2.1267357
2: -10.2043324, -7.3152981, -10.1443129, -7.3213940, -2.4601426, 2.4276657
3: -12.7920551, -10.3471174, -12.7821140, -10.3563156, -2.0294290, 2.0281458
4: 5.8710923, 8.4850378, 5.8858681, 8.4309330, -2.3201571, 2.3279858
5: -8.3798885, -5.7470326, -8.3676147, -5.7517195, -2.0590577, 2.0466623
6: -12.9543877, -9.7024460, -12.7108364, -9.7072382, -2.3593340, 2.3107977
7: -6.2408209, -3.3283496, -6.2174826, -3.3342159, -2.8157859, 2.7987375
8: -3.0490851, -0.2229128, -3.0022950, -0.2282777, -2.3448329, 2.3003302
9: -5.4763236, -3.1923084, -5.4689345, -3.2161679, -1.7405963, 1.7590675

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109443, upper bound: 1.1242815
time: 5.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255760, upper bound: 1.1255780
time: 6.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.24 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.24
Output dim: 4, lower bound: -1.1003205, upper bound: 1.1242446
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.24
Output dim: 4, lower bound: -1.1147428, upper bound: 1.1255307
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.24
Output dim: 4, lower bound: -1.1109443, upper bound: 1.1242815
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.24
Output dim: 4, lower bound: -1.1255760, upper bound: 1.1255780

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.2583046, -11.1325111, -14.2390966, -11.1000738, -2.5635862, 2.5126872
1: -10.6033821, -7.9200602, -10.5854397, -7.9352579, -2.0712533, 2.0387464
2: -10.1361475, -7.3592310, -10.1238670, -7.3668509, -2.3560677, 2.3430364
3: -12.7639837, -10.3640976, -12.7486420, -10.3735142, -1.9776583, 1.9781890
4: 5.9173641, 8.4049873, 5.9281902, 8.3748255, -2.2149472, 2.1750603
5: -8.3531055, -5.7692533, -8.3379498, -5.7840567, -1.9943485, 1.9903784
6: -12.7042570, -9.7974138, -12.6933012, -9.7621365, -2.2625184, 2.1255591
7: -6.1831017, -3.3422990, -6.1525049, -3.3663425, -2.5853953, 2.7191033
8: -2.9935217, -0.2501640, -2.9854293, -0.2508950, -2.2615175, 2.2422352
9: -5.4293261, -3.2258496, -5.3925037, -3.2499232, -1.5322852, 1.6528785

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1190572
time: 4.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1242318
time: 5.22 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.2714787, -11.1257534, -14.2720432, -11.0880489, -2.5823431, 2.5736785
1: -10.6126785, -7.9058170, -10.6156960, -7.9030695, -2.1116166, 2.1258326
2: -10.1411486, -7.3414044, -10.1435680, -7.3261147, -2.4074235, 2.3985105
3: -12.7772465, -10.3578892, -12.7809620, -10.3566999, -2.0249376, 2.0162702
4: 5.9066386, 8.4296675, 5.8907623, 8.4306278, -2.2661076, 2.3013639
5: -8.3625040, -5.7561893, -8.3664141, -5.7527795, -2.0351229, 2.0414455
6: -12.7085752, -9.7807646, -12.7103062, -9.7245464, -2.3012013, 2.2518723
7: -6.2112126, -3.3355637, -6.2159891, -3.3345447, -2.8006401, 2.7890587
8: -2.9995480, -0.2421093, -3.0016418, -0.2315316, -2.2906446, 2.2829657
9: -5.4626217, -3.2182455, -5.4674344, -3.2166567, -1.7273216, 1.7079558

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136627, upper bound: 1.1109206
time: 5.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136627, upper bound: 1.1109219
time: 5.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.4110470, -11.0689564, -14.2392693, -11.0884476, -2.6143889, 2.5753944
1: -10.6145916, -7.9066291, -10.5863552, -7.9344082, -2.0835476, 2.0530839
2: -10.1993103, -7.3332157, -10.1246004, -7.3621545, -2.4070110, 2.3745871
3: -12.7788076, -10.3533154, -12.7497883, -10.3731194, -1.9933186, 1.9895868
4: 5.8819218, 8.4603662, 5.9232969, 8.3751202, -2.2484980, 2.1919904
5: -8.3705235, -5.7600889, -8.3391514, -5.7830052, -2.0181084, 2.0005383
6: -12.9500618, -9.7190781, -12.6938286, -9.7448320, -2.3149829, 2.1844196
7: -6.2127357, -3.3350711, -6.1539936, -3.3660169, -2.6146250, 2.7281313
8: -3.0430560, -0.2310009, -2.9860792, -0.2476478, -2.3145027, 2.2596173
9: -5.4430614, -3.1999035, -5.3939848, -3.2494378, -1.5431900, 1.6791201

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109291, upper bound: 1.1191040
time: 5.03 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109291, upper bound: 1.1242687
time: 6.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.4242506, -11.0621958, -14.2722225, -11.0764294, -2.6367574, 2.6363873
1: -10.6239052, -7.8924031, -10.6166172, -7.9022236, -2.1239614, 2.1401551
2: -10.2043304, -7.3152990, -10.1443090, -7.3214025, -2.4583211, 2.4299693
3: -12.7920570, -10.3471184, -12.7821054, -10.3563175, -2.0401812, 2.0276401
4: 5.8710942, 8.4850368, 5.8858728, 8.4309235, -2.2997398, 2.3139553
5: -8.3798866, -5.7470322, -8.3676100, -5.7517252, -2.0588217, 2.0515752
6: -12.9543877, -9.7024498, -12.7108345, -9.7072468, -2.3528867, 2.3107934
7: -6.2408180, -3.3283513, -6.2174711, -3.3342190, -2.8299470, 2.7980714
8: -3.0490847, -0.2229123, -3.0022922, -0.2282801, -2.3436370, 2.3003273
9: -5.4763212, -3.1923089, -5.4689174, -3.2161694, -1.7381227, 1.7313919

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242813, upper bound: 1.1109424
time: 4.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242813, upper bound: 1.1109455
time: 5.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1190572
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1242318
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1136627, upper bound: 1.1109206
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1136627, upper bound: 1.1109219
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1109291, upper bound: 1.1191040
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1109291, upper bound: 1.1242687
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1242813, upper bound: 1.1109424
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 4, lower bound: -1.1242813, upper bound: 1.1109455

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2483482, -11.1408291, -14.2371368, -11.1017780, -2.5480123, 2.4990554
1: -10.5983658, -7.9298611, -10.5838566, -7.9368887, -2.0646501, 2.0267229
2: -10.0983171, -7.3807683, -10.1074467, -7.3672700, -2.3177605, 2.3021374
3: -12.7449131, -10.3779449, -12.7411461, -10.3746204, -1.9574232, 1.9568031
4: 5.9436741, 8.3837643, 5.9313030, 8.3645458, -2.1778312, 2.1509304
5: -8.3478031, -5.7812777, -8.3371305, -5.7858148, -1.9838314, 1.9767349
6: -12.6933680, -9.8028851, -12.6920681, -9.7637911, -2.2497163, 2.1189778
7: -6.1632662, -3.3655980, -6.1502519, -3.3766313, -2.5553894, 2.6933813
8: -2.9605145, -0.2740378, -2.9705009, -0.2517624, -2.2278013, 2.2031713
9: -5.4070358, -3.2414799, -5.3819528, -3.2505245, -1.5098767, 1.6259444

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1084439
time: 4.60 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1190561
time: 4.56 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.2582970, -11.1325130, -14.2390928, -11.1000767, -2.5651832, 2.5090408
1: -10.6033773, -7.9200640, -10.5854387, -7.9352617, -2.0711293, 2.0392971
2: -10.1361151, -7.3592315, -10.1238537, -7.3668518, -2.3248119, 2.3430171
3: -12.7639656, -10.3641005, -12.7486343, -10.3735132, -1.9657502, 1.9781811
4: 5.9173713, 8.4049654, 5.9281931, 8.3748150, -2.2149367, 2.1551394
5: -8.3531046, -5.7692566, -8.3379478, -5.7840595, -1.9929667, 1.9892991
6: -12.7042503, -9.7974167, -12.6932983, -9.7621365, -2.2625141, 2.1238775
7: -6.1830950, -3.3423195, -6.1525030, -3.3663521, -2.5787716, 2.6978412
8: -2.9934826, -0.2501669, -2.9854136, -0.2508960, -2.2341566, 2.2422190
9: -5.4293108, -3.2258506, -5.3924971, -3.2499242, -1.5077078, 1.6528711

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1136503
time: 7.14 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1242318
time: 4.65 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.2720432, -11.0880489, -2.5512629, 2.5302546
1: -10.5824461, -7.9380188, -10.6156960, -7.9030695, -2.0536849, 2.0758431
2: -10.1214695, -7.3821011, -10.1435680, -7.3261147, -2.3757286, 2.3495684
3: -12.7449169, -10.3747272, -12.7809620, -10.3566999, -1.9775410, 1.9937279
4: 5.9440694, 8.3738632, 5.8907623, 8.4306278, -2.1720347, 2.2391043
5: -8.3340082, -5.7874660, -8.3664141, -5.7527795, -1.9984827, 2.0039680
6: -12.6915712, -9.8183794, -12.7103062, -9.7245464, -2.2088594, 2.2114291
7: -6.1476965, -3.3673813, -6.2159891, -3.3345447, -2.7185688, 2.6005101
8: -2.9833360, -0.2614660, -3.0016418, -0.2315316, -2.2618718, 2.2588615
9: -5.3876762, -3.2515135, -5.4674344, -3.2166567, -1.6526814, 1.5388646

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003204, upper bound: 1.1003207
time: 6.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003204, upper bound: 1.1109195
time: 6.20 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.2720432, -11.0880489, -2.6103745, 2.5736728
1: -10.6126757, -7.9058218, -10.6156960, -7.9030695, -2.1256580, 2.1258245
2: -10.1411448, -7.3414111, -10.1435680, -7.3261147, -2.4098363, 2.3985038
3: -12.7772408, -10.3578930, -12.7809620, -10.3566999, -2.0249333, 2.0275233
4: 5.9066410, 8.4296608, 5.8907623, 8.4306278, -2.2661057, 2.2813597
5: -8.3625011, -5.7561960, -8.3664141, -5.7527795, -2.0402660, 2.0414405
6: -12.7085724, -9.7807722, -12.7103062, -9.7245464, -2.3011994, 2.2463877
7: -6.2112017, -3.3355670, -6.2159891, -3.3345447, -2.8006325, 2.8038788
8: -2.9995470, -0.2421117, -3.0016418, -0.2315316, -2.2906408, 2.2817707
9: -5.4626083, -3.2182460, -5.4674344, -3.2166567, -1.7048221, 1.7079542

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003205, upper bound: 1.1003233
time: 5.31 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003205, upper bound: 1.1109186
time: 5.04 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.4010735, -11.0772724, -14.2373104, -11.0901527, -2.5988865, 2.5618081
1: -10.6095896, -7.9164419, -10.5847721, -7.9360380, -2.0769553, 2.0410557
2: -10.1615047, -7.3548117, -10.1081781, -7.3625765, -2.3686676, 2.3337183
3: -12.7598124, -10.3671942, -12.7422953, -10.3742323, -1.9731665, 1.9681942
4: 5.9082069, 8.4391499, 5.9264045, 8.3648405, -2.2111745, 2.1678855
5: -8.3651991, -5.7721043, -8.3383369, -5.7847643, -2.0076265, 1.9868872
6: -12.9391460, -9.7245522, -12.6926022, -9.7464876, -2.3017511, 2.1778417
7: -6.1930084, -3.3583229, -6.1517420, -3.3763061, -2.5845280, 2.7024655
8: -3.0100975, -0.2548909, -2.9711504, -0.2485151, -2.2808256, 2.2205667
9: -5.4207540, -3.2155447, -5.3834372, -3.2500401, -1.5207334, 1.6487136

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109058, upper bound: 1.1084440
time: 5.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109093, upper bound: 1.1087251
time: 5.74 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4110441, -11.0689583, -14.2392702, -11.0884514, -2.6095209, 2.5717335
1: -10.6145840, -7.9066334, -10.5863533, -7.9344106, -2.0834236, 2.0536370
2: -10.1992769, -7.3332162, -10.1245842, -7.3621540, -2.3721776, 2.3704665
3: -12.7787895, -10.3533173, -12.7497787, -10.3731232, -1.9814105, 1.9895802
4: 5.8819294, 8.4603415, 5.9233007, 8.3751097, -2.2484837, 2.1696429
5: -8.3705206, -5.7600927, -8.3391514, -5.7830076, -2.0167270, 1.9994586
6: -12.9500618, -9.7190819, -12.6938276, -9.7448359, -2.3103042, 2.1827214
7: -6.2127314, -3.3350890, -6.1539927, -3.3660247, -2.6078229, 2.7068706
8: -3.0430188, -0.2310019, -2.9860611, -0.2476497, -2.2853036, 2.2596016
9: -5.4430466, -3.1999030, -5.3939791, -3.2494388, -1.5185423, 1.6670921

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109058, upper bound: 1.1136524
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109093, upper bound: 1.1136500
time: 5.75 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.2722225, -11.0764294, -2.5989280, 2.5929651
1: -10.5936718, -7.9245777, -10.6166172, -7.9022236, -2.0659640, 2.0902834
2: -10.1846275, -7.3561811, -10.1443090, -7.3214025, -2.4033246, 2.3811719
3: -12.7597809, -10.3639021, -12.7821054, -10.3563175, -1.9928451, 2.0051773
4: 5.9087772, 8.4292564, 5.8858728, 8.4309235, -2.2040348, 2.2517056
5: -8.3513088, -5.7782631, -8.3676100, -5.7517252, -2.0149062, 2.0141566
6: -12.9373684, -9.7400455, -12.7108345, -9.7072468, -2.2461247, 2.2703159
7: -6.1773005, -3.3601122, -6.2174711, -3.3342190, -2.7478895, 2.6095486
8: -3.0328588, -0.2423491, -3.0022922, -0.2282801, -2.3114257, 2.2761717
9: -5.4014525, -3.2255373, -5.4689174, -3.2161694, -1.6635387, 1.5495918

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1057563, upper bound: 1.1109295
time: 5.18 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109289, upper bound: 1.1109295
time: 4.69 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4242449, -11.0621948, -14.2722225, -11.0764294, -2.6583695, 2.6363828
1: -10.6239023, -7.8924074, -10.6166172, -7.9022236, -2.1380029, 2.1401496
2: -10.2043276, -7.3153048, -10.1443090, -7.3214025, -2.4549308, 2.4299634
3: -12.7920494, -10.3471184, -12.7821054, -10.3563175, -2.0401769, 2.0388923
4: 5.8710976, 8.4850273, 5.8858728, 8.4309235, -2.2997360, 2.2989225
5: -8.3798857, -5.7470388, -8.3676100, -5.7517252, -2.0613647, 2.0515699
6: -12.9543858, -9.7024517, -12.7108345, -9.7072468, -2.3498802, 2.3053100
7: -6.2408094, -3.3283525, -6.2174711, -3.3342190, -2.8299403, 2.8128901
8: -3.0490832, -0.2229147, -3.0022922, -0.2282801, -2.3436351, 2.2991323
9: -5.4763069, -3.1923113, -5.4689174, -3.2161694, -1.7157474, 1.7225900

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1057564, upper bound: 1.1109293
time: 5.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109290, upper bound: 1.1109293
time: 5.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1084439
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1190561
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1136503
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1242318
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003204, upper bound: 1.1003207
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003204, upper bound: 1.1109195
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003205, upper bound: 1.1003233
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1003205, upper bound: 1.1109186
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1109058, upper bound: 1.1084440
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1109093, upper bound: 1.1087251
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1109058, upper bound: 1.1136524
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1109093, upper bound: 1.1136500
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1057563, upper bound: 1.1109295
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1109289, upper bound: 1.1109295
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1057564, upper bound: 1.1109293
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 4, lower bound: -1.1109290, upper bound: 1.1109293

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2483482, -11.1408291, -14.2365837, -11.1395111, -2.5079260, 2.4956679
1: -10.5983658, -7.9298611, -10.5808601, -7.9396458, -2.0617704, 2.0236828
2: -10.0983171, -7.3807683, -10.1050425, -7.3825130, -2.3001981, 2.2959309
3: -12.7449131, -10.3779449, -12.7374239, -10.3758154, -1.9563198, 1.9531119
4: 5.9436741, 8.3837643, 5.9472036, 8.3635836, -2.1768231, 2.1348457
5: -8.3478031, -5.7812777, -8.3331795, -5.7892237, -1.9794970, 1.9711695
6: -12.6933680, -9.8028851, -12.6903410, -9.8200378, -2.1934638, 2.1176102
7: -6.1632662, -3.3655980, -6.1454391, -3.3776708, -2.5535684, 2.6882944
8: -2.9605145, -0.2740378, -2.9684033, -0.2623310, -2.2171936, 2.2014632
9: -5.4070358, -3.2414799, -5.3771210, -3.2521100, -1.5080943, 1.6210427

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003104, upper bound: 1.0960802
time: 5.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003103, upper bound: 1.1084440
time: 5.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2483482, -11.1408291, -14.3892803, -11.0759449, -2.5702586, 2.5335276
1: -10.5983658, -7.9298611, -10.5920868, -7.9262209, -2.0752497, 2.0345807
2: -10.0983171, -7.3807683, -10.1682205, -7.3566098, -2.3264360, 2.3291500
3: -12.7449131, -10.3779449, -12.7523012, -10.3650274, -1.9666061, 1.9684682
4: 5.9436741, 8.3837643, 5.9118376, 8.4189758, -2.1935344, 2.1699481
5: -8.3478031, -5.7812777, -8.3504906, -5.7800202, -1.9879928, 1.9890826
6: -12.6933680, -9.8028851, -12.9361248, -9.7416992, -2.2718239, 2.1601837
7: -6.1632662, -3.3655980, -6.1751003, -3.3703959, -2.5610237, 2.7172465
8: -2.9605145, -0.2740378, -3.0179434, -0.2432046, -2.2363024, 2.2463813
9: -5.4070358, -3.2414799, -5.3909006, -3.2261453, -1.5170047, 1.6329749

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1067133
time: 4.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1190561
time: 5.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.2582970, -11.1325130, -14.2385406, -11.1378136, -2.5251298, 2.5056534
1: -10.6033773, -7.9200640, -10.5824432, -7.9380193, -2.0682526, 2.0362611
2: -10.1361151, -7.3592315, -10.1214542, -7.3821001, -2.3072467, 2.3368258
3: -12.7639656, -10.3641005, -12.7449074, -10.3747272, -1.9646406, 1.9744899
4: 5.9173713, 8.4049654, 5.9440699, 8.3738518, -2.2139292, 2.1390924
5: -8.3531046, -5.7692566, -8.3340054, -5.7874670, -1.9886322, 1.9837229
6: -12.7042503, -9.7974167, -12.6915731, -9.8183784, -2.2062597, 2.1225114
7: -6.1830950, -3.3423195, -6.1476932, -3.3673887, -2.5770068, 2.6927605
8: -2.9934826, -0.2501669, -2.9833179, -0.2614679, -2.2235479, 2.2405128
9: -5.4293108, -3.2258506, -5.3876696, -3.2515144, -1.5059266, 1.6479728

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1012779
time: 13.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1136501
time: 6.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.2582970, -11.1325130, -14.3912411, -11.0742731, -2.5874505, 2.5400841
1: -10.6033773, -7.9200640, -10.5936680, -7.9245806, -2.0817351, 2.0471590
2: -10.1361151, -7.3592315, -10.1846123, -7.3561926, -2.3334827, 2.3544943
3: -12.7639656, -10.3641005, -12.7597713, -10.3639174, -1.9749341, 1.9898288
4: 5.9173713, 8.4049654, 5.9087811, 8.4292459, -2.2217636, 2.1720076
5: -8.3531046, -5.7692566, -8.3513050, -5.7782640, -1.9971290, 1.9996986
6: -12.7042503, -9.7974167, -12.9373627, -9.7400436, -2.2846098, 2.1628780
7: -6.1830950, -3.3423195, -6.1772947, -3.3601279, -2.5843153, 2.7217665
8: -2.9934826, -0.2501669, -3.0328298, -0.2423506, -2.2426453, 2.2720156
9: -5.4293108, -3.2258506, -5.4014454, -3.2255402, -1.5148335, 1.6598761

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003104, upper bound: 1.1118786
time: 5.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1003104, upper bound: 1.1242320
time: 8.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.2714739, -11.1257534, -2.5111952, 2.5268891
1: -10.5824461, -7.9380188, -10.6126757, -7.9058218, -2.0508027, 2.0727751
2: -10.1214695, -7.3821011, -10.1411448, -7.3414111, -2.3581457, 2.3433552
3: -12.7449169, -10.3747272, -12.7772408, -10.3578930, -1.9764442, 1.9900236
4: 5.9440694, 8.3738632, 5.9066410, 8.4296608, -2.1710873, 2.2232828
5: -8.3340082, -5.7874660, -8.3625011, -5.7561960, -1.9943805, 1.9984512
6: -12.6915712, -9.8183794, -12.7085724, -9.7807722, -2.1526594, 2.2100332
7: -6.1476965, -3.3673813, -6.2112017, -3.3355670, -2.7167530, 2.5955043
8: -2.9833360, -0.2614660, -2.9995470, -0.2421117, -2.2512617, 2.2571201
9: -5.3876762, -3.2515135, -5.4626083, -3.2182460, -1.6509070, 1.5342455

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136487, upper bound: 1.0951101
time: 4.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136487, upper bound: 1.1003075
time: 7.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.4242411, -11.0621967, -2.5734797, 2.5650125
1: -10.5824461, -7.9380188, -10.6239023, -7.8924084, -2.0641747, 2.0837328
2: -10.1214695, -7.3821011, -10.2043285, -7.3153167, -2.3845491, 2.3825514
3: -12.7449169, -10.3747272, -12.7920494, -10.3471355, -1.9866643, 2.0057006
4: 5.9440694, 8.3738632, 5.8710966, 8.4850273, -2.1779499, 2.2553167
5: -8.3340082, -5.7874660, -8.3798828, -5.7470398, -2.0017972, 2.0161977
6: -12.6915712, -9.8183794, -12.9543839, -9.7024555, -2.2229986, 2.2396319
7: -6.1476965, -3.3673813, -6.2408066, -3.3283591, -2.7241869, 2.6235049
8: -2.9833360, -0.2614660, -3.0490723, -0.2229152, -2.2704325, 2.3068357
9: -5.3876762, -3.2515135, -5.4763064, -3.1923122, -1.6598110, 1.5444951

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136487, upper bound: 1.1057438
time: 4.29 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136487, upper bound: 1.1109064
time: 6.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.2714739, -11.1257534, -2.5703068, 2.5703073
1: -10.6126757, -7.9058218, -10.6126757, -7.9058218, -2.1227555, 2.1227558
2: -10.1411448, -7.3414111, -10.1411448, -7.3414111, -2.3922901, 2.3922901
3: -12.7772408, -10.3578930, -12.7772408, -10.3578930, -2.0238361, 2.0238361
4: 5.9066410, 8.4296608, 5.9066410, 8.4296608, -2.2651029, 2.2651024
5: -8.3625011, -5.7561960, -8.3625011, -5.7561960, -2.0359240, 2.0359240
6: -12.7085724, -9.7807722, -12.7085724, -9.7807722, -2.2449927, 2.2449925
7: -6.2112017, -3.3355670, -6.2112017, -3.3355670, -2.7988176, 2.7988176
8: -2.9995470, -0.2421117, -2.9995470, -0.2421117, -2.2800303, 2.2800293
9: -5.4626083, -3.2182460, -5.4626083, -3.2182460, -1.7030549, 1.7030549

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1094784
time: 5.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1147293
time: 6.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257534, -14.4242411, -11.0621967, -2.6325908, 2.6058478
1: -10.6126757, -7.9058218, -10.6239023, -7.8924084, -2.1361232, 2.1337142
2: -10.1411448, -7.3414111, -10.2043285, -7.3153167, -2.4185781, 2.4320080
3: -12.7772408, -10.3578930, -12.7920494, -10.3471355, -2.0340571, 2.0391154
4: 5.9066410, 8.4296608, 5.8710966, 8.4850273, -2.2775936, 2.3015223
5: -8.3625011, -5.7561960, -8.3798828, -5.7470398, -2.0443625, 2.0536704
6: -12.7085724, -9.7807722, -12.9543839, -9.7024555, -2.3233752, 2.2762685
7: -6.2112017, -3.3355670, -6.2408066, -3.3283591, -2.8062506, 2.8278103
8: -2.9995470, -0.2421117, -3.0490723, -0.2229152, -2.2991781, 2.3297453
9: -5.4626083, -3.2182460, -5.4763064, -3.1923122, -1.7161527, 1.7149000

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1202934
time: 5.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1255184
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.4010735, -11.0772724, -14.2365837, -11.1395111, -2.5463743, 2.5579829
1: -10.6095896, -7.9164419, -10.5808601, -7.9396458, -2.0727024, 2.0370727
2: -10.1615047, -7.3548117, -10.1050425, -7.3825130, -2.3457294, 2.3222332
3: -12.7598124, -10.3671942, -12.7374239, -10.3758154, -1.9720993, 1.9633667
4: 5.9082069, 8.4391499, 5.9472036, 8.3635836, -2.2129607, 2.1466954
5: -8.3651991, -5.7721043, -8.3331795, -5.7892237, -1.9973440, 1.9796190
6: -12.9391460, -9.7245522, -12.6903410, -9.8200378, -2.2281213, 2.1959486
7: -6.1930084, -3.3583229, -6.1454391, -3.3776708, -2.5823922, 2.6958008
8: -3.0100975, -0.2548909, -2.9684033, -0.2623310, -2.2669544, 2.2206597
9: -5.4207540, -3.2155447, -5.3771210, -3.2521100, -1.5182767, 1.6422744

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109056, upper bound: 1.0960801
time: 5.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109056, upper bound: 1.1084466
time: 4.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.4010735, -11.0772724, -14.3892822, -11.0759459, -2.5779657, 2.5657630
1: -10.6095896, -7.9164419, -10.5920877, -7.9262195, -2.0867901, 2.0485787
2: -10.1615047, -7.3548117, -10.1682215, -7.3565993, -2.3487558, 2.3445735
3: -12.7598124, -10.3671942, -12.7523012, -10.3650131, -1.9776039, 1.9739292
4: 5.9082069, 8.4391499, 5.9118376, 8.4189758, -2.2262616, 2.1783154
5: -8.3651991, -5.7721043, -8.3504925, -5.7800207, -2.0116072, 2.0021856
6: -12.9391460, -9.7245522, -12.9361277, -9.7417011, -2.2700353, 2.1940765
7: -6.1930084, -3.3583229, -6.1751032, -3.3703883, -2.5904942, 2.7253952
8: -3.0100975, -0.2548909, -3.0179553, -0.2432051, -2.2570066, 2.2412782
9: -5.4207540, -3.2155447, -5.3909006, -3.2261448, -1.5273173, 1.6422133

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109068, upper bound: 1.0960794
time: 5.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109069, upper bound: 1.1084435
time: 5.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4110441, -11.0689583, -14.2385406, -11.1378136, -2.5570478, 2.5679440
1: -10.6145840, -7.9066334, -10.5824432, -7.9380193, -2.0791750, 2.0496583
2: -10.1992769, -7.3332162, -10.1214542, -7.3821001, -2.3492451, 2.3608847
3: -12.7787895, -10.3533173, -12.7449074, -10.3747272, -1.9803343, 1.9847505
4: 5.8819294, 8.4603415, 5.9440699, 8.3738518, -2.2466688, 2.1484923
5: -8.3705206, -5.7600927, -8.3340054, -5.7874670, -2.0064445, 1.9921772
6: -12.9500618, -9.7190819, -12.6915731, -9.8183784, -2.2366834, 2.2008309
7: -6.2127314, -3.3350890, -6.1476932, -3.3673887, -2.6051159, 2.7002149
8: -3.0430188, -0.2310019, -2.9833179, -0.2614679, -2.2714233, 2.2596960
9: -5.4430466, -3.1999030, -5.3876696, -3.2515144, -1.5160937, 1.6606591

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109056, upper bound: 1.1012805
time: 5.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109058, upper bound: 1.1136525
time: 5.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4110441, -11.0689583, -14.3912458, -11.0742741, -2.5952015, 2.5756860
1: -10.6145840, -7.9066334, -10.5936680, -7.9245796, -2.0932655, 2.0611649
2: -10.1992769, -7.3332162, -10.1846132, -7.3561802, -2.3558960, 2.3813601
3: -12.7787895, -10.3533173, -12.7597704, -10.3639021, -1.9858475, 1.9952953
4: 5.8819294, 8.4603415, 5.9087801, 8.4292450, -2.2548003, 2.1802337
5: -8.3705206, -5.7600927, -8.3513069, -5.7782650, -2.0207100, 2.0101602
6: -12.9500618, -9.7190819, -12.9373636, -9.7400484, -2.2828560, 2.1989658
7: -6.2127314, -3.3350890, -6.1772985, -3.3601201, -2.6136608, 2.7298613
8: -3.0430188, -0.2310019, -3.0328403, -0.2423496, -2.2633114, 2.2802863
9: -5.4430466, -3.1999030, -5.4014463, -3.2255383, -1.5251248, 1.6691201

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109093, upper bound: 1.1012786
time: 6.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109069, upper bound: 1.1136490
time: 5.29 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.3892822, -11.0759459, -14.2622004, -11.0847874, -2.5853195, 2.5773065
1: -10.5920877, -7.9262195, -10.6116409, -7.9120178, -2.0539432, 2.0837033
2: -10.1682215, -7.3565993, -10.1064987, -7.3430166, -2.3590798, 2.3428960
3: -12.7523012, -10.3650131, -12.7630405, -10.3702650, -1.9714222, 1.9849415
4: 5.9118376, 8.4189758, 5.9123602, 8.4096985, -2.1798272, 2.2099309
5: -8.3504925, -5.7800207, -8.3623238, -5.7637300, -2.0012679, 2.0035861
6: -12.9361277, -9.7417011, -12.6999578, -9.7126846, -2.2396107, 2.2575333
7: -6.1751032, -3.3703883, -6.1976223, -3.3574998, -2.7221179, 2.5754118
8: -3.0179553, -0.2432051, -2.9692883, -0.2521830, -2.2663946, 2.2424650
9: -5.3909006, -3.2261448, -5.4466519, -3.2318234, -1.6331143, 1.5271778

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1190614, upper bound: 1.1049920
time: 4.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1190960, upper bound: 1.1109229
time: 5.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.66 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003104, upper bound: 1.0960802
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003103, upper bound: 1.1084440
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1067133
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003078, upper bound: 1.1190561
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1012779
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003077, upper bound: 1.1136501
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003104, upper bound: 1.1118786
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1003104, upper bound: 1.1242320
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1136487, upper bound: 1.0951101
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1136487, upper bound: 1.1003075
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1136487, upper bound: 1.1057438
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1136487, upper bound: 1.1109064
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1094784
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1147293
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1202934
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1018620, upper bound: 1.1255184
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109056, upper bound: 1.0960801
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109056, upper bound: 1.1084466
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109068, upper bound: 1.0960794
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109069, upper bound: 1.1084435
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109056, upper bound: 1.1012805
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109058, upper bound: 1.1136525
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109093, upper bound: 1.1012786
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1109069, upper bound: 1.1136490
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1190614, upper bound: 1.1049920
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.66
Output dim: 4, lower bound: -1.1190960, upper bound: 1.1109229
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 4, lower bound: -1.1109289, upper bound: 1.1109295
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 4, lower bound: -1.1057564, upper bound: 1.1109293
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 4, lower bound: -1.1109290, upper bound: 1.1109293
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.308117389678955
rel_dist={4: [-1.1255949748543852, 1.1255974909157898]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013866, upper bound: 1.0090144
time: 6.13 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090140, upper bound: 1.0090172
time: 5.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.40 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.40
Output dim: 4, lower bound: -1.0013866, upper bound: 1.0090144
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.40
Output dim: 4, lower bound: -1.0090140, upper bound: 1.0090172

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2714777, -11.1257515, -14.2719250, -11.0961533, -2.4445653, 2.4157262
1: -10.6126785, -7.9058170, -10.6150503, -7.9036565, -2.0211101, 2.0212426
2: -10.1411486, -7.3414025, -10.1430531, -7.3293958, -2.3109674, 2.3020666
3: -12.7772465, -10.3578882, -12.7801666, -10.3569536, -1.9398084, 1.9418392
4: 5.9066377, 8.4296684, 5.8941717, 8.4304304, -2.2279058, 2.2398763
5: -8.3625040, -5.7561874, -8.3655796, -5.7535124, -1.9512935, 1.9522226
6: -12.7085762, -9.7807646, -12.7099342, -9.7366228, -2.1826043, 2.1395752
7: -6.2112141, -3.3355632, -6.2149668, -3.3347614, -2.7170362, 2.7195802
8: -2.9995489, -0.2421093, -3.0011969, -0.2338033, -2.2197523, 2.2127924
9: -5.4626245, -3.2182441, -5.4664121, -3.2169943, -1.6653032, 1.6677594

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906212, upper bound: 1.0081376
time: 10.18 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013777, upper bound: 1.0090071
time: 6.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.4242506, -11.0621929, -14.2722263, -11.0764313, -2.5048852, 2.4705606
1: -10.6239071, -7.8924012, -10.6166210, -7.9022169, -2.0340610, 2.0362222
2: -10.2043324, -7.3152981, -10.1443119, -7.3213978, -2.3655810, 2.3313887
3: -12.7920551, -10.3471174, -12.7821121, -10.3563166, -1.9550467, 1.9540029
4: 5.8710923, 8.4850378, 5.8858709, 8.4309311, -2.2598190, 2.2690902
5: -8.3798885, -5.7470326, -8.3676138, -5.7517219, -1.9757214, 1.9635327
6: -12.9543877, -9.7024460, -12.7108364, -9.7072411, -2.2460971, 2.1864042
7: -6.2408209, -3.3283496, -6.2174816, -3.3342164, -2.7465754, 2.7296853
8: -3.0490851, -0.2229128, -3.0022955, -0.2282796, -2.2750263, 2.2290726
9: -5.4763236, -3.1923084, -5.4689341, -3.2161698, -1.6756997, 1.6944263

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982685, upper bound: 1.0081430
time: 7.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090050, upper bound: 1.0090063
time: 5.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 4, lower bound: -0.9906212, upper bound: 1.0081376
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 4, lower bound: -1.0013777, upper bound: 1.0090071
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 4, lower bound: -0.9982685, upper bound: 1.0081430
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 4, lower bound: -1.0090050, upper bound: 1.0090063

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -14.2556734, -11.1332321, -14.2389755, -11.1081867, -2.4205365, 2.3823524
1: -10.6016445, -7.9228296, -10.5848007, -7.9358521, -1.9451497, 1.9441881
2: -10.1351852, -7.3626909, -10.1233530, -7.3701305, -2.2433739, 2.2432919
3: -12.7614117, -10.3651428, -12.7478371, -10.3737755, -1.9009900, 1.8943586
4: 5.9190722, 8.4001560, 5.9316053, 8.3746204, -2.0980172, 2.1044278
5: -8.3513489, -5.7718101, -8.3371048, -5.7847948, -1.9066882, 1.9031911
6: -12.7035742, -9.8006477, -12.6929264, -9.7742195, -2.0590582, 2.0032096
7: -6.1776581, -3.3436246, -6.1514668, -3.3665671, -2.5094995, 2.5052247
8: -2.9924474, -0.2517223, -2.9849820, -0.2531686, -2.1753273, 2.1693592
9: -5.4228277, -3.2268767, -5.3914633, -3.2502646, -1.4612963, 1.4589453

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906144, upper bound: 1.0042726
time: 15.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0081300
time: 4.61 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -14.2714758, -11.1257524, -14.2719212, -11.0961552, -2.4401560, 2.4392738
1: -10.6126766, -7.9058180, -10.6150494, -7.9036627, -2.0199375, 2.0340672
2: -10.1411476, -7.3414054, -10.1430511, -7.3294048, -2.3107595, 2.3042672
3: -12.7772455, -10.3578892, -12.7801590, -10.3569546, -1.9500871, 1.9409022
4: 5.9066386, 8.4296684, 5.8941770, 8.4304199, -2.2064676, 2.2387400
5: -8.3625050, -5.7561903, -8.3655758, -5.7535186, -1.9508576, 1.9569175
6: -12.7085762, -9.7807674, -12.7099304, -9.7366295, -2.1768441, 2.1395705
7: -6.2112088, -3.3355639, -6.2149544, -3.3347635, -2.7305727, 2.7183437
8: -2.9995489, -0.2421098, -3.0011950, -0.2338061, -2.2184982, 2.2127876
9: -5.4626184, -3.2182455, -5.4663963, -3.2169967, -1.6623001, 1.6416695

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013680, upper bound: 1.0051283
time: 5.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013682, upper bound: 1.0089999
time: 5.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.4084129, -11.0696783, -14.2392693, -11.0884514, -2.4788475, 2.4371948
1: -10.6128464, -7.9093981, -10.5863562, -7.9344110, -1.9580410, 1.9591794
2: -10.1983433, -7.3366904, -10.1246004, -7.3621559, -2.2980762, 2.2727149
3: -12.7762375, -10.3543816, -12.7497864, -10.3731213, -1.9166532, 1.9065671
4: 5.8836541, 8.4555359, 5.9233003, 8.3751183, -2.1288347, 2.1273551
5: -8.3687305, -5.7626438, -8.3391504, -5.7830057, -1.9311886, 1.9145460
6: -12.9493837, -9.7223091, -12.6938276, -9.7448378, -2.1254621, 2.0499806
7: -6.2072954, -3.3363905, -6.1539927, -3.3660173, -2.5389657, 2.5153461
8: -3.0419850, -0.2325640, -2.9860792, -0.2476501, -2.2305360, 2.1856527
9: -5.4365683, -3.2009258, -5.3939838, -3.2494378, -1.4716735, 1.4857082

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982589, upper bound: 1.0042813
time: 5.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982591, upper bound: 1.0081318
time: 5.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.4242487, -11.0621958, -14.2722225, -11.0764341, -2.5019145, 2.4941149
1: -10.6239052, -7.8924036, -10.6166182, -7.9022245, -2.0328889, 2.0490496
2: -10.2043304, -7.3153019, -10.1443081, -7.3214040, -2.3623338, 2.3335893
3: -12.7920542, -10.3471184, -12.7821054, -10.3563194, -1.9653249, 1.9530649
4: 5.8710928, 8.4850330, 5.8858757, 8.4309225, -2.2383809, 2.2550588
5: -8.3798866, -5.7470355, -8.3676090, -5.7517276, -1.9752874, 1.9682274
6: -12.9543867, -9.7024498, -12.7108364, -9.7072506, -2.2386260, 2.1863995
7: -6.2408161, -3.3283503, -6.2174702, -3.3342180, -2.7601118, 2.7284484
8: -3.0490851, -0.2229133, -3.0022917, -0.2282844, -2.2737718, 2.2290673
9: -5.4763184, -3.1923094, -5.4689174, -3.2161708, -1.6725695, 1.6633055

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089956, upper bound: 1.0051317
time: 5.86 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089956, upper bound: 1.0089981
time: 4.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.08 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 26.08
Output dim: 4, lower bound: -0.9906144, upper bound: 1.0042726
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.08
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0081300
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 26.08
Output dim: 4, lower bound: -1.0013680, upper bound: 1.0051283
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.08
Output dim: 4, lower bound: -1.0013682, upper bound: 1.0089999
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 26.08
Output dim: 4, lower bound: -0.9982589, upper bound: 1.0042813
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.08
Output dim: 4, lower bound: -0.9982591, upper bound: 1.0081318
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.08
Output dim: 4, lower bound: -1.0089956, upper bound: 1.0051317
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.08
Output dim: 4, lower bound: -1.0089956, upper bound: 1.0089981

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.2556686, -11.1332331, -14.2389774, -11.1081886, -2.4217529, 2.3782992
1: -10.6016378, -7.9228344, -10.5847960, -7.9358549, -1.9449959, 1.9447124
2: -10.1351500, -7.3626924, -10.1233349, -7.3701301, -2.2101917, 2.2432656
3: -12.7613935, -10.3651428, -12.7478294, -10.3737783, -1.8884897, 1.8943505
4: 5.9190812, 8.4001322, 5.9316111, 8.3746071, -2.0980053, 2.0835128
5: -8.3513470, -5.7718148, -8.3371038, -5.7847958, -1.9050326, 1.9018216
6: -12.7035704, -9.8006516, -12.6929274, -9.7742252, -2.0590529, 2.0014334
7: -6.1776528, -3.3436439, -6.1514626, -3.3665786, -2.5039988, 2.4829063
8: -2.9924097, -0.2517242, -2.9849601, -0.2531700, -2.1465983, 2.1693387
9: -5.4228144, -3.2268763, -5.3914557, -3.2502651, -1.4353790, 1.4589360

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0005039
time: 4.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0081300
time: 4.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257563, -14.2719164, -11.0961571, -2.4413614, 2.4352179
1: -10.6126709, -7.9058232, -10.6150455, -7.9036655, -2.0197849, 2.0345895
2: -10.1411133, -7.3414073, -10.1430302, -7.3294039, -2.2775707, 2.3042428
3: -12.7772284, -10.3578920, -12.7801495, -10.3569555, -1.9375877, 1.9408937
4: 5.9066467, 8.4296436, 5.8941822, 8.4304075, -2.2032795, 2.2148652
5: -8.3625011, -5.7561936, -8.3655739, -5.7535200, -1.9492030, 1.9551294
6: -12.7085714, -9.7807713, -12.7099285, -9.7366314, -2.1768389, 2.1378448
7: -6.2112026, -3.3355842, -6.2149520, -3.3347743, -2.7231803, 2.6960201
8: -2.9995103, -0.2421122, -3.0011725, -0.2338095, -2.1897683, 2.2127681
9: -5.4626040, -3.2182455, -5.4663887, -3.2169981, -1.6363709, 1.6394453

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013682, upper bound: 1.0013689
time: 6.41 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013682, upper bound: 1.0089999
time: 5.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4084063, -11.0696840, -14.2392683, -11.0884523, -2.4735999, 2.4331260
1: -10.6128397, -7.9094028, -10.5863533, -7.9344115, -1.9578867, 1.9597020
2: -10.1983080, -7.3366895, -10.1245804, -7.3621578, -2.2611418, 2.2681291
3: -12.7762203, -10.3543816, -12.7497768, -10.3731232, -1.9041524, 1.9065578
4: 5.8836617, 8.4555130, 5.9233041, 8.3751087, -2.1288214, 2.1035204
5: -8.3687286, -5.7626467, -8.3391514, -5.7830067, -1.9295330, 1.9131768
6: -12.9493809, -9.7223167, -12.6938295, -9.7448387, -2.1207814, 2.0481853
7: -6.2072887, -3.3364110, -6.1539879, -3.3660271, -2.5332994, 2.4930224
8: -3.0419440, -0.2325659, -2.9860573, -0.2476511, -2.1988602, 2.1856332
9: -5.4365520, -3.2009277, -5.3939767, -3.2494392, -1.4456823, 1.4736810

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9940332, upper bound: 1.0081274
time: 5.05 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982543, upper bound: 1.0081272
time: 5.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.4142160, -11.0705109, -14.2698078, -11.0784731, -2.4859171, 2.4801140
1: -10.6189327, -7.9022174, -10.6147652, -7.9041519, -2.0260119, 2.0366535
2: -10.1665249, -7.3369656, -10.1249189, -7.3219280, -2.3239021, 2.2892714
3: -12.7730579, -10.3610401, -12.7732420, -10.3576593, -1.9451365, 1.9303398
4: 5.8974581, 8.4638138, 5.8897057, 8.4187622, -2.1991158, 2.2301340
5: -8.3746309, -5.7590332, -8.3667135, -5.7538214, -1.9642630, 1.9544311
6: -12.9434738, -9.7079058, -12.7093983, -9.7091799, -2.2251191, 2.1796598
7: -6.2210865, -3.3515911, -6.2148046, -3.3463550, -2.7282209, 2.7023501
8: -3.0161288, -0.2468281, -2.9846649, -0.2293143, -2.2399364, 2.1873198
9: -5.4540215, -3.2079616, -5.4564619, -3.2168913, -1.6500432, 1.6314554

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9974229, upper bound: 1.0047057
time: 6.12 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089899, upper bound: 1.0051248
time: 5.17 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4242458, -11.0621986, -14.2722158, -11.0764380, -2.4966116, 2.4900446
1: -10.6239014, -7.8924084, -10.6166143, -7.9022245, -2.0327353, 2.0495732
2: -10.2042961, -7.3153009, -10.1442881, -7.3214064, -2.3253589, 2.3310506
3: -12.7920361, -10.3471184, -12.7820959, -10.3563204, -1.9528246, 1.9530563
4: 5.8711033, 8.4850101, 5.8858795, 8.4309101, -2.2349906, 2.2311668
5: -8.3798847, -5.7470388, -8.3676090, -5.7517304, -1.9712567, 1.9664972
6: -12.9543858, -9.7024536, -12.7108316, -9.7072506, -2.2339554, 2.1846557
7: -6.2408118, -3.3283696, -6.2174673, -3.3342309, -2.7525840, 2.7061253
8: -3.0490460, -0.2229147, -3.0022697, -0.2282844, -2.2402949, 2.2290483
9: -5.4763031, -3.1923113, -5.4689083, -3.2161713, -1.6465678, 1.6512697

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974229, upper bound: 1.0085559
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089899, upper bound: 1.0089949
time: 5.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.78 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.78
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0005039
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0081300
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 24.78
Output dim: 4, lower bound: -1.0013682, upper bound: 1.0013689
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -1.0013682, upper bound: 1.0089999
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -0.9940332, upper bound: 1.0081274
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -0.9982543, upper bound: 1.0081272
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 24.78
Output dim: 4, lower bound: -0.9974229, upper bound: 1.0047057
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -1.0089899, upper bound: 1.0051248
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -0.9974229, upper bound: 1.0085559
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.78
Output dim: 4, lower bound: -1.0089899, upper bound: 1.0089949

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.2556686, -11.1332331, -14.3912411, -11.0742741, -2.4526320, 2.4086630
1: -10.6016378, -7.9228344, -10.5936661, -7.9245806, -1.9561367, 1.9532247
2: -10.1351500, -7.3626924, -10.1846085, -7.3561926, -2.2226787, 2.2556357
3: -12.7613935, -10.3651428, -12.7597704, -10.3639164, -1.8979111, 1.9071910
4: 5.9190812, 8.4001322, 5.9087825, 8.4292412, -2.1061196, 2.1031170
5: -8.3513470, -5.7718148, -8.3513050, -5.7782650, -1.9101310, 1.9138172
6: -12.7035704, -9.8006516, -12.9373598, -9.7400484, -2.0848546, 2.0432825
7: -6.1776528, -3.3436439, -6.1772957, -3.3601286, -2.5099416, 2.5078111
8: -2.9924097, -0.2517242, -3.0328257, -0.2423506, -2.1574340, 2.1985726
9: -5.4228144, -3.2268763, -5.4014425, -3.2255406, -1.4428854, 1.4670496

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906117, upper bound: 0.9996930
time: 5.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0081300
time: 4.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.2714739, -11.1257563, -14.4242382, -11.0622005, -2.4722452, 2.4633224
1: -10.6126709, -7.9058232, -10.6238995, -7.8924098, -2.0308728, 2.0431402
2: -10.1411133, -7.3414073, -10.2043095, -7.3153186, -2.2900834, 2.3184910
3: -12.7772284, -10.3578920, -12.7920399, -10.3471365, -1.9469461, 1.9532800
4: 5.9066467, 8.4296436, 5.8711019, 8.4850159, -2.2092676, 2.2320786
5: -8.3625011, -5.7561936, -8.3798828, -5.7470398, -1.9542360, 1.9648044
6: -12.7085714, -9.7807713, -12.9543839, -9.7024574, -2.1979570, 2.1627233
7: -6.2112026, -3.3355842, -6.2408042, -3.3283691, -2.7291470, 2.7210460
8: -2.9995103, -0.2421122, -3.0490503, -0.2229166, -2.2005868, 2.2404585
9: -5.4626040, -3.2182455, -5.4762979, -3.1923141, -1.6438708, 1.6451042

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906117, upper bound: 0.9982472
time: 4.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906119, upper bound: 0.9982451
time: 5.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4005041, -11.0772076, -14.2238626, -11.1032343, -2.4682941, 2.4263558
1: -10.5841408, -7.9126267, -10.5325165, -7.9533229, -1.8840499, 1.8834531
2: -10.1896076, -7.3569131, -10.1005678, -7.4020004, -2.2039313, 2.2015650
3: -12.7626553, -10.3594780, -12.7203064, -10.3855782, -1.7903504, 1.7881129
4: 5.8922176, 8.4377689, 5.9471817, 8.3412523, -1.7589085, 1.7246959
5: -8.3552999, -5.7683210, -8.3116179, -5.7977724, -1.7751198, 1.7527122
6: -12.9416084, -9.7438936, -12.6728973, -9.7835407, -1.9211588, 1.8630271
7: -6.1973944, -3.3615296, -6.1260767, -3.4143696, -2.3708010, 2.3326592
8: -3.0098205, -0.2408042, -2.9223418, -0.2752380, -2.1297708, 2.1101718
9: -5.4316673, -3.2067366, -5.3843536, -3.2612824, -1.2289310, 1.2565979

Time for backsubstitution: 15.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9940305, upper bound: 0.9996998
time: 6.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9940307, upper bound: 1.0081270
time: 7.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4084091, -11.0696850, -14.2392664, -11.0884590, -2.4728990, 2.4406896
1: -10.6128378, -7.9094033, -10.5863457, -7.9344139, -1.9577706, 1.9189255
2: -10.1983070, -7.3366928, -10.1245794, -7.3621635, -2.2390723, 2.2615855
3: -12.7762184, -10.3543835, -12.7497711, -10.3731222, -1.9037609, 1.8935409
4: 5.8836632, 8.4555101, 5.9233074, 8.3751011, -2.1245813, 2.0965767
5: -8.3687267, -5.7626486, -8.3391428, -5.7830114, -1.9275305, 1.9130514
6: -12.9493790, -9.7223186, -12.6938257, -9.7448502, -2.1103969, 2.0481820
7: -6.2072878, -3.3364129, -6.1539850, -3.3660347, -2.4994688, 2.4924884
8: -3.0419409, -0.2325659, -2.9860463, -0.2476549, -2.1863980, 2.1549850
9: -5.4365520, -3.2009282, -5.3939748, -3.2494421, -1.4441011, 1.4804893

Time for backsubstitution: 15.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982542, upper bound: 0.9997002
time: 13.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982543, upper bound: 1.0081272
time: 6.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.4136829, -11.0705147, -14.2688751, -11.0784760, -2.4845080, 2.4817500
1: -10.6189318, -7.9022341, -10.6147633, -7.9041834, -2.0250759, 2.0390940
2: -10.1665249, -7.3369818, -10.1249199, -7.3219547, -2.3208337, 2.2910414
3: -12.7729988, -10.3610420, -12.7731428, -10.3576603, -1.9464211, 1.9297643
4: 5.8974638, 8.4638090, 5.8897133, 8.4187555, -2.1936622, 2.2202830
5: -8.3746281, -5.7590628, -8.3667088, -5.7538757, -1.9637713, 1.9550200
6: -12.9434319, -9.7079086, -12.7093239, -9.7091875, -2.2194648, 2.1796112
7: -6.2210822, -3.3516150, -6.2147951, -3.3463960, -2.7203407, 2.7021680
8: -3.0161109, -0.2468300, -2.9846325, -0.2293186, -2.2375054, 2.1872945
9: -5.4540133, -3.2079630, -5.4564495, -3.2168937, -1.6369317, 1.6180453

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0009066
time: 4.21 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0051204
time: 5.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.4220982, -11.0636482, -14.2670031, -11.0784140, -2.4880013, 2.4836116
1: -10.6218386, -7.8956461, -10.6091967, -7.9081511, -2.0246353, 2.0362780
2: -10.2034264, -7.3200779, -10.1388006, -7.3302693, -2.3138041, 2.3168652
3: -12.7881641, -10.3486042, -12.7743683, -10.3609447, -1.9433336, 1.9437318
4: 5.8750000, 8.4771786, 5.9056253, 8.4174957, -2.2140279, 2.1994677
5: -8.3776093, -5.7494431, -8.3608780, -5.7565055, -1.9640179, 1.9559679
6: -12.9532547, -9.7101889, -12.6998777, -9.7205095, -2.2184176, 2.1655889
7: -6.2301235, -3.3295419, -6.1991806, -3.3483970, -2.7236676, 2.6863246
8: -3.0474551, -0.2270894, -2.9952769, -0.2360373, -2.2296886, 2.2173553
9: -5.4645615, -3.1951165, -5.4486609, -3.2357664, -1.6130512, 1.6292641

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9974183, upper bound: 1.0043504
time: 6.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974183, upper bound: 1.0085520
time: 5.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.4237165, -11.0622005, -14.2713032, -11.0764389, -2.4952087, 2.4916801
1: -10.6238976, -7.8924255, -10.6166096, -7.9022560, -2.0317984, 2.0520129
2: -10.2042961, -7.3153172, -10.1442871, -7.3214321, -2.3222914, 2.3285444
3: -12.7919807, -10.3471212, -12.7819948, -10.3563232, -1.9541106, 1.9524796
4: 5.8711071, 8.4850073, 5.8858900, 8.4309006, -2.2244382, 2.2213058
5: -8.3798828, -5.7470670, -8.3676043, -5.7517834, -1.9695859, 1.9647841
6: -12.9543428, -9.7024555, -12.7107620, -9.7072601, -2.2282987, 2.1846068
7: -6.2408056, -3.3283935, -6.2174587, -3.3342712, -2.7426300, 2.7059426
8: -3.0490274, -0.2229185, -3.0022373, -0.2282891, -2.2378645, 2.2290230
9: -5.4762955, -3.1923122, -5.4688940, -3.2161741, -1.6334579, 1.6378591

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0047769
time: 6.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0089877
time: 8.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 36.22 seconds
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9906117, upper bound: 0.9996930
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9906117, upper bound: 1.0081300
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9906117, upper bound: 0.9982472
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9906119, upper bound: 0.9982451
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9940305, upper bound: 0.9996998
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9940307, upper bound: 1.0081270
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9982542, upper bound: 0.9997002
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9982543, upper bound: 1.0081272
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0009066
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0051204
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9974183, upper bound: 1.0043504
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -0.9974183, upper bound: 1.0085520
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0047769
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 36.22
Output dim: 4, lower bound: -1.0089852, upper bound: 1.0089877

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.2712526, -11.1262722, -14.3912411, -11.0742741, -2.4577665, 2.4071949
1: -10.6113367, -7.9060645, -10.5936661, -7.9245806, -1.9634786, 1.9709916
2: -10.1407423, -7.3415365, -10.1846085, -7.3561926, -2.2284436, 2.2638257
3: -12.7770405, -10.3595905, -12.7597704, -10.3639164, -1.9138317, 1.9111965
4: 5.9106579, 8.4295940, 5.9087825, 8.4292412, -2.0974169, 2.1095905
5: -8.3616514, -5.7563190, -8.3513050, -5.7782650, -1.9198108, 1.9173048
6: -12.7068472, -9.7808571, -12.9373598, -9.7400484, -2.0752535, 2.0482569
7: -6.2105527, -3.3358686, -6.1772957, -3.3601286, -2.5154524, 2.5143147
8: -2.9982748, -0.2422738, -3.0328257, -0.2423506, -2.1639228, 2.2057428
9: -5.4622622, -3.2230506, -5.4014425, -3.2255406, -1.4471006, 1.4530284

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9863641, upper bound: 1.0081206
time: 4.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906069, upper bound: 1.0081229
time: 4.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4161196, -11.0702209, -14.2238626, -11.1032343, -2.4732828, 2.4278145
1: -10.5937862, -7.8958712, -10.5325165, -7.9533229, -1.8914099, 1.9012222
2: -10.1950798, -7.3356524, -10.1005678, -7.4020004, -2.2021804, 2.2104416
3: -12.7783213, -10.3541241, -12.7203064, -10.3855782, -1.8067136, 1.7924736
4: 5.8840079, 8.4672203, 5.9471817, 8.3412523, -1.7509954, 1.7307711
5: -8.3656149, -5.7530508, -8.3116179, -5.7977724, -1.7781277, 1.7618020
6: -12.9448605, -9.7244921, -12.6728973, -9.7835407, -1.9120970, 1.8869917
7: -6.2301021, -3.3537657, -6.1260767, -3.4143696, -2.3760962, 2.3351679
8: -3.0157084, -0.2315116, -2.9223418, -0.2752380, -2.1337242, 2.1222987
9: -5.4708662, -3.2029190, -5.3843536, -3.2612824, -1.2333188, 1.2422445

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9940066, upper bound: 1.0005023
time: 17.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9940077, upper bound: 1.0004988
time: 4.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4240198, -11.0627117, -14.2392664, -11.0884590, -2.4777632, 2.4420705
1: -10.6225615, -7.8926430, -10.5863457, -7.9344139, -1.9651649, 1.9366877
2: -10.2039280, -7.3154364, -10.1245794, -7.3621635, -2.2375813, 2.2698400
3: -12.7918463, -10.3488197, -12.7497711, -10.3731222, -1.9196568, 1.8975072
4: 5.8751001, 8.4849577, 5.9233074, 8.3751011, -2.1160173, 2.1030593
5: -8.3790321, -5.7471595, -8.3391428, -5.7830114, -1.9307206, 1.9172485
6: -12.9526577, -9.7025375, -12.6938257, -9.7448502, -2.1008148, 2.0810139
7: -6.2401905, -3.3286581, -6.1539850, -3.3660347, -2.5049586, 2.4969273
8: -3.0478077, -0.2230797, -2.9860463, -0.2476549, -2.1905451, 2.1676860
9: -5.4759812, -3.1971169, -5.3939748, -3.2494421, -1.4483516, 1.4658906

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982387, upper bound: 1.0005023
time: 5.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982401, upper bound: 1.0004986
time: 4.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.3985224, -11.0853653, -14.2609959, -11.0859632, -2.4791217, 2.4595299
1: -10.5650530, -7.9210529, -10.5859728, -7.9074674, -1.9490433, 1.9891603
2: -10.1422987, -7.3767700, -10.1160545, -7.3421359, -2.2547197, 2.2362595
3: -12.7435846, -10.3735790, -12.7595692, -10.3629332, -1.9121704, 1.8107567
4: 5.9214258, 8.4299355, 5.8986635, 8.4010420, -1.8274713, 2.1766033
5: -8.3469162, -5.7739868, -8.3532963, -5.7597170, -1.9260092, 1.8019681
6: -12.9224663, -9.7471819, -12.7015009, -9.7310781, -2.1712923, 1.9948874
7: -6.1930051, -3.3999643, -6.2049532, -3.3715119, -2.5563745, 2.6415124
8: -2.9524078, -0.2745481, -2.9525666, -0.2376804, -2.1639543, 2.1244922
9: -5.4444666, -3.2197332, -5.4512863, -3.2227163, -1.4301205, 1.5953624

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089847, upper bound: 0.9932700
time: 4.25 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089858, upper bound: 0.9932759
time: 4.30 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.4136810, -11.0705166, -14.2688751, -11.0784779, -2.4904780, 2.4810615
1: -10.6189232, -7.9022355, -10.6147594, -7.9041848, -1.9843140, 2.0389729
2: -10.1665220, -7.3369856, -10.1249180, -7.3219547, -2.3142905, 2.2645845
3: -12.7729950, -10.3610439, -12.7731390, -10.3576622, -1.9334092, 1.9293613
4: 5.8974676, 8.4638014, 5.8897157, 8.4187546, -2.1876726, 2.2142811
5: -8.3746223, -5.7590656, -8.3667078, -5.7538743, -1.9619336, 1.9540298
6: -12.9434299, -9.7079220, -12.7093220, -9.7091923, -2.2153296, 2.1699610
7: -6.2210784, -3.3516226, -6.2147956, -3.3463991, -2.7112842, 2.6659808
8: -3.0161004, -0.2468338, -2.9846292, -0.2293196, -2.2047772, 2.1872883
9: -5.4540114, -3.2079663, -5.4564481, -3.2168946, -1.6437905, 1.6164405

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089847, upper bound: 0.9974993
time: 5.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089858, upper bound: 0.9974993
time: 4.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4220963, -11.0636501, -14.2670040, -11.0784149, -2.4939752, 2.4829245
1: -10.6218309, -7.8956466, -10.6091938, -7.9081531, -1.9838691, 2.0361574
2: -10.2034245, -7.3200827, -10.1388006, -7.3302717, -2.3072605, 2.2939000
3: -12.7881603, -10.3486042, -12.7743673, -10.3609457, -1.9303207, 1.9433231
4: 5.8750033, 8.4771709, 5.9056273, 8.4174919, -2.2071962, 2.1934733
5: -8.3776035, -5.7494445, -8.3608780, -5.7565069, -1.9621804, 1.9530303
6: -12.9532490, -9.7102013, -12.6998806, -9.7205124, -2.2142839, 2.1559377
7: -6.2301226, -3.3295488, -6.1991806, -3.3483996, -2.7146125, 2.6501441
8: -3.0474448, -0.2270923, -2.9952722, -0.2360373, -2.1969619, 2.2172389
9: -5.4645619, -3.1951199, -5.4486618, -3.2357683, -1.6199200, 1.6276605

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9974124, upper bound: 1.0008981
time: 6.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9974135, upper bound: 1.0008979
time: 4.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.4084778, -11.0770397, -14.2634125, -11.0839300, -2.4898138, 2.4695120
1: -10.5699558, -7.9112477, -10.5878239, -7.9055281, -1.9558458, 2.0020735
2: -10.1800632, -7.3551311, -10.1354465, -7.3416166, -2.2557545, 2.2721922
3: -12.7626705, -10.3598652, -12.7684240, -10.3616028, -1.9199591, 1.8333127
4: 5.8954163, 8.4511366, 5.8948631, 8.4131861, -1.8579066, 2.1776333
5: -8.3522310, -5.7622266, -8.3541870, -5.7576351, -1.9318542, 1.8097456
6: -12.9333458, -9.7419968, -12.7029343, -9.7291222, -2.1801653, 1.9998705
7: -6.2123938, -3.3767114, -6.2075005, -3.3593869, -2.5783200, 2.6452317
8: -2.9853997, -0.2508440, -2.9701662, -0.2366509, -2.1643863, 2.1645675
9: -5.4665160, -3.2040873, -5.4637356, -3.2219968, -1.4268026, 1.6151881

Time for backsubstitution: 14.65 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2409.37 seconds
