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
execution time: IAR + LP analysis = 15.90 + 33.02 = 48.92 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.08 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.4249300956726074
rel_dist={4: [-1.3292876093097172, 1.3292852400204573]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.132899761199951
rel_dist={4: [-0.7676993968362122, 0.7676969143316699]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.191305637359619
rel_dist={4: [-0.8893816350203041, 0.8893789546935258]}

## Binary Search Result
Binary search time: 214.30 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3336.78 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941729, upper bound: 1.4186513
time: 4.80 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212993, upper bound: 1.4213004
time: 7.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.72 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 12.72
Output dim: 4, lower bound: -1.3941729, upper bound: 1.4186513
IS_B2, status: Status.UNKNOWN, split count: 1, time: 12.72
Output dim: 4, lower bound: -1.4212993, upper bound: 1.4213004

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -14.2643585, -11.0804882, -14.2392712, -11.0884342, -2.9817257, 2.9653020
1: -10.6110935, -7.9107890, -10.5863619, -7.9344053, -2.3507686, 2.3223877
2: -10.1413212, -7.3321533, -10.1246023, -7.3621345, -2.6502028, 2.6593063
3: -12.7741175, -10.3600121, -12.7497921, -10.3731194, -2.2096519, 2.2050104
4: 5.8922434, 8.4160643, 5.9232817, 8.3751202, -2.4155049, 2.3767614
5: -8.3620148, -5.7595801, -8.3391552, -5.7829981, -2.2561803, 2.2530224
6: -12.7082787, -9.7172499, -12.6938314, -9.7448015, -2.6192222, 2.5605881
7: -6.2005591, -3.3382335, -6.1540017, -3.3660154, -2.8139973, 2.8157682
8: -2.9987082, -0.2331223, -2.9860821, -0.2476349, -2.4790049, 2.4721203
9: -5.4488859, -3.2206340, -5.3939929, -3.2494354, -1.7504487, 1.8508401

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941526, upper bound: 1.4028994
time: 5.27 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941549, upper bound: 1.4186325
time: 5.37 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -14.2722282, -11.0764151, -14.2722225, -11.0764179, -2.9978437, 3.0281830
1: -10.6166239, -7.9022126, -10.6166220, -7.9022179, -2.3887091, 2.4039085
2: -10.1443129, -7.3213749, -10.1443119, -7.3213830, -2.6994853, 2.7021029
3: -12.7821178, -10.3563156, -12.7821121, -10.3563175, -2.2539616, 2.2417819
4: 5.8858538, 8.4309330, 5.8858571, 8.4309235, -2.4659758, 2.4833322
5: -8.3676176, -5.7517138, -8.3676128, -5.7517204, -2.2932968, 2.2988677
6: -12.7108383, -9.7072067, -12.7108374, -9.7072144, -2.6571898, 2.6618519
7: -6.2174892, -3.3342149, -6.2174792, -3.3342180, -2.8832712, 2.8832643
8: -3.0022964, -0.2282643, -3.0022945, -0.2282662, -2.5058012, 2.5068135
9: -5.4689426, -3.2161665, -5.4689250, -3.2161689, -1.9282265, 1.9074702

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053688, upper bound: 1.4212820
time: 7.31 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212802, upper bound: 1.4212824
time: 9.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.59 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 31.59
Output dim: 4, lower bound: -1.3941526, upper bound: 1.4028994
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 31.59
Output dim: 4, lower bound: -1.3941549, upper bound: 1.4186325
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 31.59
Output dim: 4, lower bound: -1.4053688, upper bound: 1.4212820
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 31.59
Output dim: 4, lower bound: -1.4212802, upper bound: 1.4212824

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -14.2643585, -11.0804882, -14.2385406, -11.1378107, -2.9292722, 2.9608645
1: -10.6110935, -7.9107890, -10.5824461, -7.9380188, -2.3469925, 2.3184052
2: -10.1413212, -7.3321533, -10.1214695, -7.3821011, -2.6272173, 2.6511936
3: -12.7741175, -10.3600121, -12.7449169, -10.3747272, -2.2081881, 2.2001760
4: 5.8922434, 8.4160643, 5.9440694, 8.3738632, -2.4141884, 2.3557014
5: -8.3620148, -5.7595801, -8.3340082, -5.7874660, -2.2505012, 2.2457352
6: -12.7082787, -9.7172499, -12.6915712, -9.8183794, -2.5456228, 2.5587931
7: -6.2005591, -3.3382335, -6.1476965, -3.3673813, -2.8115988, 2.8094630
8: -2.9987082, -0.2331223, -2.9833360, -0.2614660, -2.4651213, 2.4698787
9: -5.4488859, -3.2206340, -5.3876762, -3.2515135, -1.7481136, 1.8444276

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3851200, upper bound: 1.4028774
time: 4.94 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941300, upper bound: 1.4028768
time: 4.63 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -14.2643557, -11.0804977, -14.3912497, -11.0742731, -3.0167131, 3.0018682
1: -10.6110916, -7.9107924, -10.5936718, -7.9245777, -2.3604774, 2.3298435
2: -10.1413193, -7.3321643, -10.1846275, -7.3561811, -2.6673250, 2.6916869
3: -12.7741165, -10.3600111, -12.7597809, -10.3639021, -2.2184935, 2.2158508
4: 5.8922529, 8.4160633, 5.9087772, 8.4292564, -2.4367876, 2.3935876
5: -8.3620129, -5.7595830, -8.3513088, -5.7782631, -2.2589989, 2.2688363
6: -12.7082748, -9.7172728, -12.9373684, -9.7400455, -2.6412516, 2.5986538
7: -6.2005553, -3.3382349, -6.1773005, -3.3601122, -2.8190451, 2.8390656
8: -2.9987068, -0.2331324, -3.0328588, -0.2423491, -2.4862437, 2.5195208
9: -5.4488802, -3.2206345, -5.4014525, -3.2255373, -1.7570348, 1.8575950

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3851180, upper bound: 1.4186104
time: 4.28 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941298, upper bound: 1.4186097
time: 5.16 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -14.2714777, -11.1257515, -14.2722225, -11.0764179, -2.9934330, 2.9757628
1: -10.6126785, -7.9058170, -10.6166220, -7.9022179, -2.3846841, 2.4001009
2: -10.1411486, -7.3414025, -10.1443119, -7.3213830, -2.6913486, 2.6791425
3: -12.7772465, -10.3578882, -12.7821121, -10.3563175, -2.2491322, 2.2403355
4: 5.9066377, 8.4296684, 5.8858571, 8.4309235, -2.4446926, 2.4820213
5: -8.3625040, -5.7561874, -8.3676128, -5.7517204, -2.2860856, 2.2931788
6: -12.7085762, -9.7807646, -12.7108374, -9.7072144, -2.6553559, 2.5883112
7: -6.2112141, -3.3355632, -6.2174792, -3.3342180, -2.8769960, 2.8819160
8: -2.9995489, -0.2421093, -3.0022945, -0.2282662, -2.5035124, 2.4929247
9: -5.4626245, -3.2182441, -5.4689250, -3.2161689, -1.9217920, 1.9051602

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3961294, upper bound: 1.4212604
time: 5.47 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053459, upper bound: 1.4212598
time: 7.18 seconds

## BFS IS instance: IS_B2_A2

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4121008, upper bound: 1.4212606
time: 5.49 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212573, upper bound: 1.4212602
time: 4.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.66 seconds
IS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.3851200, upper bound: 1.4028774
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.3941300, upper bound: 1.4028768
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.3851180, upper bound: 1.4186104
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.3941298, upper bound: 1.4186097
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.3961294, upper bound: 1.4212604
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.4053459, upper bound: 1.4212598
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.4121008, upper bound: 1.4212606
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.66
Output dim: 4, lower bound: -1.4212573, upper bound: 1.4212602

## BFS IS instance: IS_B1_B1_B1

### Backsubstitution after applying IS history:
0: -14.2631645, -11.0814972, -14.2286644, -11.1461411, -2.9163027, 2.9463072
1: -10.6101656, -7.9117484, -10.5773792, -7.9478159, -2.3357136, 2.3124611
2: -10.1316185, -7.3324099, -10.0836191, -7.4035726, -2.5938649, 2.6130712
3: -12.7696905, -10.3606739, -12.7258101, -10.3885622, -2.1897788, 2.1804998
4: 5.8941388, 8.4099865, 5.9702740, 8.3526268, -2.3910913, 2.3212416
5: -8.3615608, -5.7606192, -8.3286247, -5.7994957, -2.2371020, 2.2363329
6: -12.7075558, -9.7182198, -12.6807117, -9.8238678, -2.5395017, 2.5468249
7: -6.1992297, -3.3443024, -6.1278739, -3.3907146, -2.7867661, 2.7835715
8: -2.9898930, -0.2336359, -2.9503045, -0.2853184, -2.4322534, 2.4364886
9: -5.4426546, -3.2209916, -5.3653774, -3.2671280, -1.7218814, 1.8222437

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B1_B1_B1_A1

### Relational analysis result of IS_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4028767
time: 5.13 seconds

## Relational analysis of IS_B1_B1_B1_A2

### Relational analysis result of IS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693605, upper bound: 1.4028771
time: 5.01 seconds

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -14.2643557, -11.0804901, -14.2385397, -11.1378145, -2.9266167, 2.9635956
1: -10.6110935, -7.9107890, -10.5824404, -7.9380198, -2.3476300, 2.3183610
2: -10.1413174, -7.3321533, -10.1214342, -7.3820987, -2.6272097, 2.6257110
3: -12.7741146, -10.3600121, -12.7448978, -10.3747292, -2.2081842, 2.1900449
4: 5.8922458, 8.4160595, 5.9440761, 8.3738384, -2.3972416, 2.3451443
5: -8.3620138, -5.7595806, -8.3340034, -5.7874694, -2.2501307, 2.2451754
6: -12.7082767, -9.7172480, -12.6915693, -9.8183794, -2.5441527, 2.5587897
7: -6.2005577, -3.3382363, -6.1476908, -3.3673999, -2.7935276, 2.8094544
8: -2.9987020, -0.2331238, -2.9832954, -0.2614679, -2.4651136, 2.4466209
9: -5.4488840, -3.2206349, -5.3876624, -3.2515154, -1.7361197, 1.8247812

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B1_B1_B2_A1

### Relational analysis result of IS_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4028782
time: 4.68 seconds

## Relational analysis of IS_B1_B1_B2_A2

### Relational analysis result of IS_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4028767
time: 4.42 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -14.2631617, -11.0815058, -14.3813486, -11.0826035, -3.0037899, 2.9873576
1: -10.6101608, -7.9117517, -10.5886211, -7.9343839, -2.3491948, 2.3239117
2: -10.1316166, -7.3324208, -10.1468029, -7.3777137, -2.6340036, 2.6535101
3: -12.7696857, -10.3606739, -12.7407513, -10.3777542, -2.2000785, 2.1962538
4: 5.8941474, 8.4099884, 5.9349623, 8.4080296, -2.4136972, 2.3585174
5: -8.3615580, -5.7606239, -8.3459930, -5.7902865, -2.2455945, 2.2593784
6: -12.7075539, -9.7182426, -12.9264708, -9.7455387, -2.6351285, 2.5862501
7: -6.1992269, -3.3443031, -6.1575851, -3.3834004, -2.7942386, 2.8132820
8: -2.9898930, -0.2336445, -2.9998751, -0.2661953, -2.4533873, 2.4861674
9: -5.4426494, -3.2209930, -5.3791370, -3.2411623, -1.7307754, 1.8353660

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B1_B2_B1_A1

### Relational analysis result of IS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4186110
time: 6.87 seconds

## Relational analysis of IS_B1_B2_B1_A2

### Relational analysis result of IS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4040299
time: 4.86 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -14.2643538, -11.0804996, -14.3912430, -11.0742779, -3.0140433, 2.9981909
1: -10.6110907, -7.9107919, -10.5936651, -7.9245815, -2.3611159, 2.3297992
2: -10.1413145, -7.3321638, -10.1845932, -7.3561811, -2.6673169, 2.6631513
3: -12.7741137, -10.3600130, -12.7597628, -10.3639030, -2.2184896, 2.2057190
4: 5.8922548, 8.4160576, 5.9087849, 8.4292326, -2.4188929, 2.3827167
5: -8.3620138, -5.7595835, -8.3513060, -5.7782669, -2.2586269, 2.2655511
6: -12.7082729, -9.7172718, -12.9373627, -9.7400494, -2.6397638, 2.5939860
7: -6.2005529, -3.3382370, -6.1772947, -3.3601322, -2.8009739, 2.8390577
8: -2.9987011, -0.2331333, -3.0328188, -0.2423525, -2.4862366, 2.4962626
9: -5.4488783, -3.2206340, -5.4014378, -3.2255387, -1.7450190, 1.8378384

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4186093
time: 6.41 seconds

## Relational analysis of IS_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4040299
time: 4.84 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2702770, -11.1267586, -14.2622004, -11.0847740, -2.9804559, 2.9610016
1: -10.6117496, -7.9067764, -10.6116457, -7.9120131, -2.3734484, 2.3942313
2: -10.1314430, -7.3416586, -10.1065016, -7.3429956, -2.6579504, 2.6410160
3: -12.7728167, -10.3585510, -12.7630444, -10.3702641, -2.2305789, 2.2207017
4: 5.9085374, 8.4235926, 5.9123468, 8.4096985, -2.4216013, 2.4461770
5: -8.3620481, -5.7572327, -8.3623295, -5.7637234, -2.2727180, 2.2836902
6: -12.7078571, -9.7817354, -12.6999645, -9.7126513, -2.6493206, 2.5762794
7: -6.2098818, -3.3416309, -6.1976309, -3.3574986, -2.8523831, 2.8559999
8: -2.9907365, -0.2426219, -2.9692898, -0.2521682, -2.4706001, 2.4595175
9: -5.4563918, -3.2186003, -5.4466600, -3.2318225, -1.8955421, 1.8830025

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3961294, upper bound: 1.4053470
time: 4.69 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3961294, upper bound: 1.4212604
time: 5.07 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2714787, -11.1257534, -14.2722187, -11.0764208, -2.9908142, 2.9785454
1: -10.6126766, -7.9058175, -10.6166162, -7.9022236, -2.3853226, 2.4000559
2: -10.1411438, -7.3414021, -10.1442766, -7.3213854, -2.6913419, 2.6537204
3: -12.7772455, -10.3578892, -12.7820940, -10.3563194, -2.2491293, 2.2302039
4: 5.9066381, 8.4296646, 5.8858647, 8.4309025, -2.4277458, 2.4705024
5: -8.3625040, -5.7561898, -8.3676128, -5.7517233, -2.2857156, 2.2926183
6: -12.7085743, -9.7807665, -12.7108364, -9.7072172, -2.6540003, 2.5883074
7: -6.2112122, -3.3355680, -6.2174730, -3.3342385, -2.8769736, 2.8819051
8: -2.9995441, -0.2421098, -3.0022545, -0.2282686, -2.5035067, 2.4696674
9: -5.4626217, -3.2182426, -5.4689093, -3.2161694, -1.9097648, 1.8855152

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053459, upper bound: 1.4053492
time: 9.46 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4053459, upper bound: 1.4212598
time: 7.29 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.4230461, -11.0631847, -14.2621984, -11.0847826, -3.0260534, 3.0484359
1: -10.6229782, -7.8933692, -10.6116419, -7.9120164, -2.3849456, 2.4075773
2: -10.1946373, -7.3155560, -10.1065016, -7.3430109, -2.7060132, 2.6809754
3: -12.7876339, -10.3477840, -12.7630405, -10.3702631, -2.2462029, 2.2309265
4: 5.8729477, 8.4789610, 5.9123554, 8.4096966, -2.4606667, 2.4529219
5: -8.3794355, -5.7480721, -8.3623257, -5.7637291, -2.2956824, 2.2921281
6: -12.9536676, -9.7034149, -12.6999598, -9.7126751, -2.6883078, 2.6719508
7: -6.2395239, -3.3344097, -6.1976256, -3.3574989, -2.8820250, 2.8632159
8: -3.0402894, -0.2234268, -2.9692898, -0.2521787, -2.5203390, 2.4806991
9: -5.4700942, -3.1926713, -5.4466543, -3.2318239, -1.9085097, 1.9097090

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3851166, upper bound: 1.3941300
time: 4.64 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3851168, upper bound: 1.3941321
time: 4.94 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.4242487, -11.0621967, -14.2722158, -11.0764303, -3.0341444, 3.0660102
1: -10.6239052, -7.8924007, -10.6166134, -7.9022250, -2.3968201, 2.4134240
2: -10.2043247, -7.3152971, -10.1442766, -7.3213973, -2.7249584, 2.6936851
3: -12.7920532, -10.3471174, -12.7820902, -10.3563194, -2.2647433, 2.2404337
4: 5.8710938, 8.4850321, 5.8858752, 8.4308996, -2.4668660, 2.4772425
5: -8.3798885, -5.7470322, -8.3676100, -5.7517285, -2.3086896, 2.3010559
6: -12.9543858, -9.7024498, -12.7108335, -9.7072392, -2.6921306, 2.6839721
7: -6.2408199, -3.3283534, -6.2174683, -3.3342373, -2.9065826, 2.8891149
8: -3.0490801, -0.2229123, -3.0022535, -0.2282777, -2.5401011, 2.4908428
9: -5.4763222, -3.1923099, -5.4689074, -3.2161713, -1.9227540, 1.9113040

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941286, upper bound: 1.3941295
time: 4.62 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3941288, upper bound: 1.4212596
time: 7.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.88 seconds
IS_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4028767
IS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3693605, upper bound: 1.4028771
IS_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4028782
IS_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4028767
IS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4186110
IS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4040299
IS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4186093
IS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3783895, upper bound: 1.4040299
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3961294, upper bound: 1.4053470
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3961294, upper bound: 1.4212604
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.4053459, upper bound: 1.4053492
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.4053459, upper bound: 1.4212598
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3851166, upper bound: 1.3941300
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3851168, upper bound: 1.3941321
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3941286, upper bound: 1.3941295
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.88
Output dim: 4, lower bound: -1.3941288, upper bound: 1.4212596

## BFS IS instance: IS_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -14.2624245, -11.1308355, -14.2286644, -11.1461411, -2.9118834, 2.8938475
1: -10.6062269, -7.9153557, -10.5773792, -7.9478159, -2.3316972, 2.3086767
2: -10.1284571, -7.3524170, -10.0836191, -7.4035726, -2.5857320, 2.5900588
3: -12.7648182, -10.3622503, -12.7258101, -10.3885622, -2.1849279, 2.1790557
4: 5.9149113, 8.4087267, 5.9702740, 8.3526268, -2.3698158, 2.3199995
5: -8.3564301, -5.7650938, -8.3286247, -5.7994957, -2.2298822, 2.2306473
6: -12.7052898, -9.7917862, -12.6807117, -9.8238678, -2.5376706, 2.4732482
7: -6.1929388, -3.3456562, -6.1278739, -3.3907146, -2.7800989, 2.7822177
8: -2.9871478, -0.2474785, -2.9503045, -0.2853184, -2.4299626, 2.4225984
9: -5.4363317, -3.2230668, -5.3653774, -3.2671280, -1.7154474, 1.8199353

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3523383, upper bound: 1.3977556
time: 4.54 seconds

## Relational analysis of IS_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_B1_B1_A1_A1

### Relational analysis result of IS_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.3783949
time: 4.57 seconds

## Relational analysis of IS_B1_B1_B1_A1_A2

### Relational analysis result of IS_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.4028768
time: 4.53 seconds

## BFS IS instance: IS_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -14.4151754, -11.0672646, -14.2286644, -11.1461411, -2.9535303, 2.9561648
1: -10.6174450, -7.9019394, -10.5773792, -7.9478159, -2.3426349, 2.3220642
2: -10.1916389, -7.3263674, -10.0836191, -7.4035726, -2.6324329, 2.6164663
3: -12.7796459, -10.3514614, -12.7258101, -10.3885622, -2.2006245, 2.1893003
4: 5.8793831, 8.4641008, 5.9702740, 8.3526268, -2.4061337, 2.3268628
5: -8.3738480, -5.7559295, -8.3286247, -5.7994957, -2.2476583, 2.2390957
6: -12.9510975, -9.7134571, -12.6807117, -9.8238678, -2.5764322, 2.5516200
7: -6.2226005, -3.3384249, -6.1278739, -3.3907146, -2.8089848, 2.7894490
8: -3.0366998, -0.2283030, -2.9503045, -0.2853184, -2.4797025, 2.4417844
9: -5.4500570, -3.1971297, -5.3653774, -3.2671280, -1.7272420, 1.8466024

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_B1_B1_A2_B1

### Relational analysis result of IS_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3523383, upper bound: 1.3977554
time: 4.55 seconds

## Relational analysis of IS_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.3783953
time: 4.43 seconds

## Relational analysis of IS_B1_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.4028768
time: 4.46 seconds

## BFS IS instance: IS_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -14.2636118, -11.1298323, -14.2385397, -11.1378145, -2.9221992, 2.9111805
1: -10.6071568, -7.9143944, -10.5824404, -7.9380198, -2.3436160, 2.3145783
2: -10.1381588, -7.3521638, -10.1214342, -7.3820987, -2.6190729, 2.6026950
3: -12.7692451, -10.3615913, -12.7448978, -10.3747292, -2.2033334, 2.1885953
4: 5.9130239, 8.4147968, 5.9440761, 8.3738384, -2.3759594, 2.3439031
5: -8.3568916, -5.7640533, -8.3340034, -5.7874694, -2.2429037, 2.2394886
6: -12.7060127, -9.7908134, -12.6915693, -9.8183794, -2.5423207, 2.4852128
7: -6.1942720, -3.3395891, -6.1476908, -3.3673999, -2.7868633, 2.8081017
8: -2.9959555, -0.2469640, -2.9832954, -0.2614679, -2.4628239, 2.4327297
9: -5.4425654, -3.2227130, -5.3876624, -3.2515154, -1.7296901, 1.8224697

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B1_B1_B2_A1_A1

### Relational analysis result of IS_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.3938461
time: 4.91 seconds

## Relational analysis of IS_B1_B1_B2_A1_A2

### Relational analysis result of IS_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4028770
time: 4.63 seconds

## BFS IS instance: IS_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -14.4163685, -11.0662785, -14.2385397, -11.1378145, -2.9615960, 2.9735007
1: -10.6183701, -7.9009724, -10.5824404, -7.9380198, -2.3545537, 2.3279676
2: -10.2013283, -7.3261127, -10.1214342, -7.3820987, -2.6513872, 2.6291010
3: -12.7840614, -10.3507977, -12.7448978, -10.3747292, -2.2190199, 2.1988442
4: 5.8775377, 8.4701710, 5.9440761, 8.3738384, -2.4123325, 2.3507609
5: -8.3743076, -5.7548900, -8.3340034, -5.7874694, -2.2606897, 2.2479382
6: -12.9518223, -9.7124853, -12.6915693, -9.8183794, -2.5802426, 2.5635934
7: -6.2238975, -3.3323646, -6.1476908, -3.3673999, -2.8157654, 2.8153262
8: -3.0454903, -0.2277880, -2.9832954, -0.2614679, -2.4994426, 2.4519105
9: -5.4562874, -3.1967702, -5.3876624, -3.2515154, -1.7415056, 1.8482184

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B1_B1_B2_A2_A1

### Relational analysis result of IS_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.3938457
time: 5.65 seconds

## Relational analysis of IS_B1_B1_B2_A2_A2

### Relational analysis result of IS_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4028773
time: 4.97 seconds

## BFS IS instance: IS_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -14.2624245, -11.1308355, -14.3813486, -11.0826035, -2.9741964, 2.9348321
1: -10.6062269, -7.9153557, -10.5886211, -7.9343839, -2.3451807, 2.3195910
2: -10.1284571, -7.3524170, -10.1468029, -7.3777137, -2.6119671, 2.6305270
3: -12.7648182, -10.3622503, -12.7407513, -10.3777542, -2.1952305, 2.1944759
4: 5.9149113, 8.4087267, 5.9349623, 8.4080296, -2.3923998, 2.3545852
5: -8.3564301, -5.7650938, -8.3459930, -5.7902865, -2.2383785, 2.2486186
6: -12.7052898, -9.7917862, -12.9264708, -9.7455387, -2.6160212, 2.5126431
7: -6.1929388, -3.3456562, -6.1575851, -3.3834004, -2.7875752, 2.8119290
8: -2.9871478, -0.2474785, -2.9998751, -0.2661953, -2.4490733, 2.4722857
9: -5.4363317, -3.2230668, -5.3791370, -3.2411623, -1.7243419, 1.8317952

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_B2_B1_A1_B1

### Relational analysis result of IS_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3523383, upper bound: 1.4134901
time: 4.36 seconds

## Relational analysis of IS_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_B2_B1_A1_A1

### Relational analysis result of IS_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.3941337
time: 5.44 seconds

## Relational analysis of IS_B1_B2_B1_A1_A2

### Relational analysis result of IS_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.4186107
time: 5.55 seconds

## BFS IS instance: IS_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -14.4151754, -11.0672646, -14.3813486, -11.0826035, -3.0077477, 2.9896679
1: -10.6174450, -7.9019394, -10.5886211, -7.9343839, -2.3567789, 2.3336394
2: -10.1916389, -7.3263674, -10.1468029, -7.3777137, -2.6447191, 2.6491647
3: -12.7796459, -10.3514614, -12.7407513, -10.3777542, -2.2068520, 2.2006450
4: 5.8793831, 8.4641008, 5.9349623, 8.4080296, -2.4284973, 2.3614488
5: -8.3738480, -5.7559295, -8.3459930, -5.7902865, -2.2625389, 2.2630210
6: -12.9510975, -9.7134571, -12.9264708, -9.7455387, -2.6514373, 2.5868945
7: -6.2226005, -3.3384249, -6.1575851, -3.3834004, -2.8175774, 2.8191602
8: -3.0366998, -0.2283030, -2.9998751, -0.2661953, -2.4741135, 2.4667721
9: -5.4500570, -3.1971297, -5.3791370, -3.2411623, -1.7361360, 1.8437474

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_B2_B1_A2_B1

### Relational analysis result of IS_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3523383, upper bound: 1.3989087
time: 5.54 seconds

## Relational analysis of IS_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_B2_B1_A2_A1

### Relational analysis result of IS_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693605, upper bound: 1.3795542
time: 5.37 seconds

## Relational analysis of IS_B1_B2_B1_A2_A2

### Relational analysis result of IS_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.4028766
time: 5.01 seconds

## BFS IS instance: IS_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -14.2636118, -11.1298323, -14.3912430, -11.0742779, -2.9844871, 2.9457047
1: -10.6071568, -7.9143944, -10.5936651, -7.9245815, -2.3571038, 2.3254819
2: -10.1381588, -7.3521638, -10.1845932, -7.3561811, -2.6453204, 2.6401689
3: -12.7692451, -10.3615913, -12.7597628, -10.3639030, -2.2136421, 2.2039361
4: 5.9130239, 8.4147968, 5.9087849, 8.4292326, -2.3976030, 2.3787856
5: -8.3568916, -5.7640533, -8.3513060, -5.7782669, -2.2514033, 2.2574224
6: -12.7060127, -9.7908134, -12.9373627, -9.7400494, -2.6206560, 2.5203846
7: -6.1942720, -3.3395891, -6.1772947, -3.3601322, -2.7943115, 2.8377056
8: -2.9959555, -0.2469640, -3.0328188, -0.2423525, -2.4819241, 2.4823790
9: -5.4425654, -3.2227130, -5.4014378, -3.2255387, -1.7385907, 1.8342645

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B1_B2_B2_A1_A1

### Relational analysis result of IS_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693584, upper bound: 1.4095899
time: 4.94 seconds

## Relational analysis of IS_B1_B2_B2_A1_A2

### Relational analysis result of IS_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4186099
time: 4.67 seconds

## BFS IS instance: IS_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4163685, -11.0662785, -14.3912430, -11.0742779, -3.0180097, 3.0070271
1: -10.6183701, -7.9009724, -10.5936651, -7.9245815, -2.3687029, 2.3395314
2: -10.2013283, -7.3261127, -10.1845932, -7.3561811, -2.6780372, 2.6618917
3: -12.7840614, -10.3507977, -12.7597628, -10.3639030, -2.2252526, 2.2101092
4: 5.8775377, 8.4701710, 5.9087849, 8.4292326, -2.4337702, 2.3856435
5: -8.3743076, -5.7548900, -8.3513060, -5.7782669, -2.2755733, 2.2691927
6: -12.9518223, -9.7124853, -12.9373627, -9.7400494, -2.6560774, 2.5988226
7: -6.2238975, -3.3323646, -6.1772947, -3.3601322, -2.8243299, 2.8449302
8: -3.0454903, -0.2277880, -3.0328188, -0.2423525, -2.5069485, 2.4768605
9: -5.4562874, -3.1967702, -5.4014378, -3.2255387, -1.7504053, 1.8462234

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_B1_B2_B2_A2_A1

### Relational analysis result of IS_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693581, upper bound: 1.3950042
time: 6.97 seconds

## Relational analysis of IS_B1_B2_B2_A2_A2

### Relational analysis result of IS_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693582, upper bound: 1.4028767
time: 4.43 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -14.2702770, -11.1267586, -14.2614632, -11.1340733, -2.9280539, 2.9565868
1: -10.6117496, -7.9067764, -10.6076880, -7.9156232, -2.3696311, 2.3901973
2: -10.1314430, -7.3416586, -10.1033154, -7.3629947, -2.6349916, 2.6328604
3: -12.7728167, -10.3585510, -12.7581663, -10.3718147, -2.2291560, 2.2158678
4: 5.9085374, 8.4235926, 5.9330335, 8.4084320, -2.4202852, 2.4249611
5: -8.3620481, -5.7572327, -8.3571596, -5.7681985, -2.2670302, 2.2764981
6: -12.7078571, -9.7817354, -12.6976929, -9.7862244, -2.5757666, 2.5744429
7: -6.2098818, -3.3416309, -6.1913643, -3.3588514, -2.8510303, 2.8497334
8: -2.9907365, -0.2426219, -2.9665451, -0.2660046, -2.4567194, 2.4572344
9: -5.4563918, -3.2186003, -5.4403272, -3.2338862, -1.8932304, 1.8765764

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693586, upper bound: 1.3783922
time: 4.59 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693585, upper bound: 1.3783899
time: 7.76 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -14.2702770, -11.1267586, -14.4142170, -11.0705128, -2.9903517, 2.9953232
1: -10.6117496, -7.9067764, -10.6189327, -7.9022217, -2.3829548, 2.4011707
2: -10.1314430, -7.3416586, -10.1665230, -7.3369703, -2.6612835, 2.6777806
3: -12.7728167, -10.3585510, -12.7730522, -10.3610420, -2.2393832, 2.2312241
4: 5.9085374, 8.4235926, 5.8974600, 8.4638090, -2.4429169, 2.4609859
5: -8.3620481, -5.7572327, -8.3746281, -5.7590365, -2.2754669, 2.2942808
6: -12.7078571, -9.7817354, -12.9434719, -9.7079086, -2.6541519, 2.6075983
7: -6.2098818, -3.3416309, -6.2210793, -3.3515928, -2.8582890, 2.8794484
8: -2.9907365, -0.2426219, -3.0161288, -0.2468295, -2.4758825, 2.5070014
9: -5.4563918, -3.2186003, -5.4540091, -3.2079630, -1.9021091, 1.8883767

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693586, upper bound: 1.3941323
time: 4.34 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3693585, upper bound: 1.3941323
time: 7.29 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -14.2714787, -11.1257534, -14.2714701, -11.1257610, -2.9383631, 2.9741349
1: -10.6126766, -7.9058175, -10.6126699, -7.9058266, -2.3815150, 2.3960319
2: -10.1411438, -7.3414021, -10.1411142, -7.3414116, -2.6683807, 2.6455224
3: -12.7772455, -10.3578892, -12.7772226, -10.3578930, -2.2476830, 2.2253742
4: 5.9066381, 8.4296646, 5.9066477, 8.4296379, -2.4264336, 2.4492376
5: -8.3625040, -5.7561898, -8.3624992, -5.7561975, -2.2800264, 2.2854061
6: -12.7085743, -9.7807665, -12.7085705, -9.7807779, -2.5804391, 2.5864749
7: -6.2112122, -3.3355680, -6.2111959, -3.3355861, -2.8756261, 2.8756280
8: -2.9995441, -0.2421098, -2.9995079, -0.2421131, -2.4896183, 2.4673800
9: -5.4626217, -3.2182426, -5.4625916, -3.2182479, -1.9074583, 1.8790960

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783900, upper bound: 1.3783892
time: 5.70 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783901, upper bound: 1.4053468
time: 6.11 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -14.2714787, -11.1257534, -14.4242420, -11.0622025, -3.0006380, 3.0063179
1: -10.6126766, -7.9058175, -10.6238966, -7.8924122, -2.3948836, 2.4069953
2: -10.1411438, -7.3414021, -10.2042942, -7.3153076, -2.6946807, 2.6874652
3: -12.7772455, -10.3578892, -12.7920332, -10.3471212, -2.2579150, 2.2406549
4: 5.9066381, 8.4296646, 5.8711057, 8.4850044, -2.4481220, 2.4854922
5: -8.3625040, -5.7561898, -8.3798809, -5.7470412, -2.2884674, 2.3031564
6: -12.7085743, -9.7807665, -12.9543819, -9.7024555, -2.6588097, 2.6153550
7: -6.2112122, -3.3355680, -6.2408047, -3.3283720, -2.8828402, 2.9052367
8: -2.9995441, -0.2421098, -3.0490448, -0.2229176, -2.5087695, 2.5171080
9: -5.4626217, -3.2182426, -5.4762926, -3.1923127, -1.9163437, 1.8908336

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783900, upper bound: 1.3941295
time: 6.84 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3783901, upper bound: 1.4212604
time: 6.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.3900900, -11.0752544, -14.2621984, -11.0847826, -2.9877920, 3.0012772
1: -10.5927343, -7.9255452, -10.6116419, -7.9120164, -2.3276825, 2.3559477
2: -10.1749268, -7.3564267, -10.1065016, -7.3430109, -2.6581697, 2.6318765
3: -12.7553606, -10.3645563, -12.7630405, -10.3702631, -2.1974549, 2.2067976
4: 5.9105802, 8.4231844, 5.9123554, 8.4096966, -2.3772421, 2.3906779
5: -8.3508244, -5.7792969, -8.3623257, -5.7637291, -2.2572708, 2.2540665
6: -12.9366331, -9.7410231, -12.6999598, -9.7126751, -2.5971026, 2.6314936
7: -6.1760044, -3.3661842, -6.1976256, -3.3574989, -2.8185055, 2.7997372
8: -3.0240502, -0.2428551, -2.9692898, -0.2521787, -2.4890895, 2.4565463
9: -5.3952165, -3.2258954, -5.4466543, -3.2318239, -1.8339224, 1.7371230

Time for backsubstitution: 14.92 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.483335494995117
rel_dist={4: [-1.421316393236534, 1.4213159903507133]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242889, upper bound: 1.1109486
time: 7.27 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255847, upper bound: 1.1255834
time: 6.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.04
Output dim: 4, lower bound: -1.1242889, upper bound: 1.1109486
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.04
Output dim: 4, lower bound: -1.1255847, upper bound: 1.1255834

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -14.2392712, -11.0884342, -14.2590437, -11.0831623, -2.5662041, 2.5803695
1: -10.5863619, -7.9344053, -10.6073151, -7.9164524, -2.0434761, 2.0761592
2: -10.1246023, -7.3621345, -10.1393013, -7.3392296, -2.3679075, 2.3696225
3: -12.7497921, -10.3731194, -12.7688551, -10.3625298, -1.9807830, 1.9828629
4: 5.9232817, 8.3751202, 5.8965845, 8.4062490, -2.1813722, 2.2365384
5: -8.3391552, -5.7829981, -8.3582163, -5.7647839, -1.9977736, 2.0029311
6: -12.6938314, -9.7448015, -12.7065201, -9.7238464, -2.1995678, 2.2816949
7: -6.1540017, -3.3660154, -6.1893940, -3.3409457, -2.7230749, 2.5926390
8: -2.9860821, -0.2476349, -2.9962668, -0.2363224, -2.2566624, 2.2670846
9: -5.3939929, -3.2494354, -5.4356451, -3.2237711, -1.6567054, 1.5392632

Time for backsubstitution: 15.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136627, upper bound: 1.1109204
time: 6.06 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242813, upper bound: 1.1109424
time: 4.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.2722225, -11.0764179, -14.2722263, -11.0764151, -2.6271429, 2.5991054
1: -10.6166220, -7.9022179, -10.6166248, -7.9022145, -2.1305962, 2.1165466
2: -10.1443119, -7.3213830, -10.1443138, -7.3213768, -2.4233913, 2.4209733
3: -12.7821121, -10.3563175, -12.7821178, -10.3563156, -2.0188584, 2.0301161
4: 5.8858571, 8.4309235, 5.8858538, 8.4309320, -2.3081121, 2.2876997
5: -8.3676128, -5.7517204, -8.3676186, -5.7517152, -2.0488296, 2.0436804
6: -12.7108374, -9.7072144, -12.7108402, -9.7072096, -2.3258495, 2.3203664
7: -6.2174792, -3.3342180, -6.2174888, -3.3342156, -2.7930202, 2.8078470
8: -3.0022945, -0.2282662, -3.0022969, -0.2282648, -2.2974000, 2.2962084
9: -5.4689250, -3.2161689, -5.4689393, -3.2161655, -1.7117801, 1.7343028

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255304, upper bound: 1.1147454
time: 5.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255784, upper bound: 1.1255760
time: 7.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.11 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 28.11
Output dim: 4, lower bound: -1.1136627, upper bound: 1.1109204
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 28.11
Output dim: 4, lower bound: -1.1242813, upper bound: 1.1109424
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.11
Output dim: 4, lower bound: -1.1255304, upper bound: 1.1147454
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.11
Output dim: 4, lower bound: -1.1255784, upper bound: 1.1255760

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -14.2385406, -11.1378107, -14.2588682, -11.0947971, -2.5494051, 2.5268681
1: -10.5824461, -7.9380188, -10.6063919, -7.9173021, -2.0385947, 2.0714321
2: -10.1214695, -7.3821011, -10.1385612, -7.3439536, -2.3543835, 2.3447142
3: -12.7449169, -10.3747272, -12.7677050, -10.3629112, -1.9755983, 1.9802527
4: 5.9440694, 8.3738632, 5.9014888, 8.4059534, -2.1600080, 2.2301965
5: -8.3340082, -5.7874660, -8.3570137, -5.7658429, -1.9891419, 1.9955513
6: -12.6915712, -9.8183794, -12.7059879, -9.7411776, -2.1804304, 2.2076583
7: -6.1476965, -3.3673813, -6.1879010, -3.3412712, -2.7158413, 2.5886569
8: -2.9833360, -0.2614660, -2.9956150, -0.2395878, -2.2511425, 2.2526536
9: -5.3876762, -3.2515135, -5.4341536, -3.2242594, -1.6497493, 1.5351202

Time for backsubstitution: 15.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136499, upper bound: 1.1057439
time: 6.01 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136500, upper bound: 1.1109068
time: 5.14 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -14.3912497, -11.0742731, -14.2590427, -11.0831776, -2.5996618, 2.5895820
1: -10.5936718, -7.9245777, -10.6073093, -7.9164557, -2.0508723, 2.0858672
2: -10.1846275, -7.3561811, -10.1392994, -7.3392487, -2.3962450, 2.3763177
3: -12.7597809, -10.3639021, -12.7688494, -10.3625307, -1.9909029, 1.9917021
4: 5.9087772, 8.4292564, 5.8965998, 8.4062471, -2.1924024, 2.2563257
5: -8.3513088, -5.7782631, -8.3582134, -5.7647905, -2.0119081, 2.0057466
6: -12.9373684, -9.7400455, -12.7065182, -9.7238798, -2.2403440, 2.2665448
7: -6.1773005, -3.3601122, -6.1893873, -3.3409457, -2.7451615, 2.5976834
8: -3.0328588, -0.2423491, -2.9962659, -0.2363367, -2.3040581, 2.2699642
9: -5.4014525, -3.2255373, -5.4356370, -3.2237735, -1.6607347, 1.5458484

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242687, upper bound: 1.1057598
time: 5.08 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242686, upper bound: 1.1109294
time: 4.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.2720432, -11.0880489, -14.2714787, -11.1257534, -2.5736790, 2.5823431
1: -10.6156960, -7.9030695, -10.6126785, -7.9058170, -2.1258326, 2.1116166
2: -10.1435680, -7.3261147, -10.1411486, -7.3414044, -2.3985100, 2.4074233
3: -12.7809620, -10.3566999, -12.7772465, -10.3578892, -2.0162702, 2.0249379
4: 5.8907623, 8.4306278, 5.9066386, 8.4296675, -2.3013639, 2.2661076
5: -8.3664141, -5.7527795, -8.3625040, -5.7561893, -2.0414457, 2.0351229
6: -12.7103062, -9.7245464, -12.7085752, -9.7807646, -2.2518721, 2.3012013
7: -6.2159891, -3.3345447, -6.2112126, -3.3355637, -2.7890592, 2.8006396
8: -3.0016418, -0.2315316, -2.9995480, -0.2421093, -2.2829657, 2.2906446
9: -5.4674344, -3.2166567, -5.4626217, -3.2182455, -1.7079558, 1.7273221

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255176, upper bound: 1.1094766
time: 5.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255176, upper bound: 1.1147295
time: 5.74 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -14.2722225, -11.0764294, -14.4242506, -11.0621958, -2.6363873, 2.6367574
1: -10.6166172, -7.9022236, -10.6239052, -7.8924031, -2.1401548, 2.1239619
2: -10.1443090, -7.3214025, -10.2043304, -7.3152990, -2.4299698, 2.4583211
3: -12.7821054, -10.3563175, -12.7920570, -10.3471184, -2.0276399, 2.0401812
4: 5.8858728, 8.4309235, 5.8710942, 8.4850368, -2.3139548, 2.2997394
5: -8.3676100, -5.7517252, -8.3798866, -5.7470322, -2.0515752, 2.0588219
6: -12.7108345, -9.7072468, -12.9543877, -9.7024498, -2.3107934, 2.3528867
7: -6.2174711, -3.3342190, -6.2408180, -3.3283513, -2.7980714, 2.8299470
8: -3.0022922, -0.2282801, -3.0490847, -0.2229123, -2.3003273, 2.3436370
9: -5.4689174, -3.2161694, -5.4763212, -3.1923089, -1.7313919, 1.7381225

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255656, upper bound: 1.1203664
time: 8.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255656, upper bound: 1.1255656
time: 7.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 31.36 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1136499, upper bound: 1.1057439
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1136500, upper bound: 1.1109068
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1242687, upper bound: 1.1057598
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1242686, upper bound: 1.1109294
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1255176, upper bound: 1.1094766
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1255176, upper bound: 1.1147295
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1255656, upper bound: 1.1203664
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.36
Output dim: 4, lower bound: -1.1255656, upper bound: 1.1255656

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -14.2286644, -11.1461411, -14.2568541, -11.0965118, -2.5339556, 2.5131669
1: -10.5773792, -7.9478159, -10.6048164, -7.9189348, -2.0319395, 2.0593822
2: -10.0836191, -7.4035726, -10.1221495, -7.3443856, -2.3161087, 2.3037877
3: -12.7258101, -10.3885622, -12.7602091, -10.3640146, -1.9555020, 1.9589896
4: 5.9702740, 8.3526268, 5.9046736, 8.3956718, -2.1233621, 2.2058258
5: -8.3286247, -5.7994957, -8.3562355, -5.7676115, -1.9786434, 1.9818776
6: -12.6807117, -9.8238678, -12.7047615, -9.7428207, -2.1676731, 2.2010574
7: -6.1278739, -3.3907146, -6.1856451, -3.3515456, -2.6859136, 2.5629244
8: -2.9503045, -0.2853184, -2.9806976, -0.2404552, -2.2174368, 2.2136540
9: -5.3653774, -3.2671280, -5.4236078, -3.2248640, -1.6273398, 1.5047345

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136499, upper bound: 1.0951101
time: 4.91 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136499, upper bound: 1.1057439
time: 5.13 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -14.2385397, -11.1378145, -14.2588625, -11.0947971, -2.5509877, 2.5232136
1: -10.5824404, -7.9380198, -10.6063881, -7.9173050, -2.0384724, 2.0719836
2: -10.1214342, -7.3820987, -10.1385469, -7.3439531, -2.3231220, 2.3446944
3: -12.7448978, -10.3747292, -12.7676945, -10.3629112, -1.9636893, 1.9802465
4: 5.9440761, 8.3738384, 5.9014921, 8.4059439, -2.1516471, 2.2102752
5: -8.3340034, -5.7874694, -8.3570118, -5.7658443, -1.9877601, 1.9944715
6: -12.6915693, -9.8183794, -12.7059841, -9.7411785, -2.1804256, 2.2059271
7: -6.1476908, -3.3673999, -6.1878986, -3.3412805, -2.7158275, 2.5674005
8: -2.9832954, -0.2614679, -2.9955997, -0.2395878, -2.2237797, 2.2526374
9: -5.3876624, -3.2515154, -5.4341469, -3.2242603, -1.6262808, 1.5231254

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136500, upper bound: 1.1003082
time: 5.00 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1136500, upper bound: 1.1109068
time: 4.85 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -14.3813486, -11.0826035, -14.2570295, -11.0848961, -2.5842586, 2.5759268
1: -10.5886211, -7.9343839, -10.6057348, -7.9180851, -2.0442309, 2.0738130
2: -10.1468029, -7.3777137, -10.1228867, -7.3396797, -2.3579173, 2.3354223
3: -12.7407513, -10.3777542, -12.7613564, -10.3636379, -1.9708843, 1.9704342
4: 5.9349623, 8.4080296, 5.8997788, 8.3959675, -2.1554646, 2.2319779
5: -8.3459930, -5.7902865, -8.3574381, -5.7665596, -2.0013449, 1.9920661
6: -12.9264708, -9.7455387, -12.7052946, -9.7255211, -2.2271543, 2.2599421
7: -6.1575851, -3.3834004, -6.1871328, -3.3512218, -2.7151279, 2.5719800
8: -2.9998751, -0.2661953, -2.9813495, -0.2372069, -2.2703905, 2.2309756
9: -5.3791370, -3.2411623, -5.4250927, -3.2243791, -1.6382790, 1.5154381

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242308, upper bound: 1.0951126
time: 6.22 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242320, upper bound: 1.0953926
time: 7.75 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -14.3912430, -11.0742779, -14.2590418, -11.0831776, -2.5948176, 2.5859118
1: -10.5936651, -7.9245815, -10.6073055, -7.9164572, -2.0507505, 2.0864182
2: -10.1845932, -7.3561811, -10.1392841, -7.3392496, -2.3613930, 2.3762977
3: -12.7597628, -10.3639030, -12.7688417, -10.3625317, -1.9789939, 1.9916954
4: 5.9087849, 8.4292326, 5.8966055, 8.4062386, -2.1837077, 2.2339540
5: -8.3513060, -5.7782669, -8.3582125, -5.7647905, -2.0071516, 2.0046663
6: -12.9373627, -9.7400494, -12.7065153, -9.7238808, -2.2356758, 2.2647958
7: -6.1772947, -3.3601322, -6.1893835, -3.3409553, -2.7451468, 2.5764260
8: -3.0328188, -0.2423525, -2.9962478, -0.2363367, -2.2739220, 2.2699490
9: -5.4014378, -3.2255387, -5.4356308, -3.2237749, -1.6371555, 1.5338323

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242308, upper bound: 1.1003102
time: 6.31 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1242320, upper bound: 1.1003074
time: 8.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.2620239, -11.0963974, -14.2694407, -11.1274662, -2.5580187, 2.5686102
1: -10.6107159, -7.9128675, -10.6111097, -7.9074478, -2.1192617, 2.0996094
2: -10.1057539, -7.3477211, -10.1247339, -7.3418393, -2.3602285, 2.3664529
3: -12.7618923, -10.3706379, -12.7697487, -10.3590126, -1.9962101, 2.0034900
4: 5.9172268, 8.4094028, 5.9098620, 8.4193792, -2.2595921, 2.2417250
5: -8.3611155, -5.7647848, -8.3617325, -5.7579594, -2.0308661, 2.0214720
6: -12.6994267, -9.7299852, -12.7073555, -9.7824087, -2.2390366, 2.2946858
7: -6.1961432, -3.3578243, -6.2089510, -3.3458343, -2.7590647, 2.7748957
8: -2.9686382, -0.2554312, -2.9846339, -0.2429819, -2.2492423, 2.2516012
9: -5.4451666, -3.2323070, -5.4520750, -3.2188468, -1.6855688, 1.6969287

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1248378, upper bound: 1.0938138
time: 6.65 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255109, upper bound: 1.1094723
time: 5.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.2720356, -11.0880556, -14.2714748, -11.1257534, -2.5753016, 2.5787158
1: -10.6156902, -7.9030724, -10.6126766, -7.9058199, -2.1257100, 2.1121686
2: -10.1435337, -7.3261156, -10.1411333, -7.3414035, -2.3672953, 2.4074044
3: -12.7809448, -10.3567019, -12.7772369, -10.3578911, -2.0043616, 2.0249310
4: 5.8907690, 8.4306040, 5.9066410, 8.4296579, -2.2879796, 2.2461863
5: -8.3664103, -5.7527838, -8.3625031, -5.7561908, -2.0400643, 2.0340431
6: -12.7103014, -9.7245474, -12.7085714, -9.7807693, -2.2518673, 2.2995787
7: -6.2159820, -3.3345611, -6.2112088, -3.3355737, -2.7890434, 2.7793803
8: -3.0016031, -0.2315331, -2.9995308, -0.2421112, -2.2556038, 2.2906275
9: -5.4674206, -3.2166576, -5.4626145, -3.2182460, -1.6844873, 1.7152979

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1248378, upper bound: 1.0990094
time: 7.40 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255109, upper bound: 1.1147221
time: 5.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.2622004, -11.0847874, -14.4222050, -11.0638809, -2.6207175, 2.6230519
1: -10.6116409, -7.9120178, -10.6223392, -7.8940454, -2.1335483, 2.1119554
2: -10.1064987, -7.3430166, -10.1879349, -7.3157363, -2.3916836, 2.4141130
3: -12.7630405, -10.3702650, -12.7845716, -10.3482494, -2.0075722, 2.0187445
4: 5.9123602, 8.4096985, 5.8742428, 8.4747496, -2.2721696, 2.2752633
5: -8.3623238, -5.7637300, -8.3791199, -5.7488012, -2.0409899, 2.0451543
6: -12.6999578, -9.7126846, -12.9531641, -9.7040825, -2.2979732, 2.3463430
7: -6.1976223, -3.3574998, -6.2386189, -3.3386068, -2.7680883, 2.8041782
8: -2.9692883, -0.2521830, -3.0341961, -0.2237854, -2.2666135, 2.3004229
9: -5.4466519, -3.2318234, -5.4657793, -3.1929231, -1.7090206, 1.7077038

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1248859, upper bound: 1.1045073
time: 5.68 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255589, upper bound: 1.1203612
time: 7.49 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.2722158, -11.0764351, -14.4242477, -11.0621977, -2.6380358, 2.6296787
1: -10.6166124, -7.9022255, -10.6239023, -7.8924050, -2.1400332, 2.1245131
2: -10.1442738, -7.3214049, -10.2043161, -7.3153009, -2.3987694, 2.4394519
3: -12.7820892, -10.3563213, -12.7920475, -10.3471193, -2.0157299, 2.0401742
4: 5.8858805, 8.4308968, 5.8710966, 8.4850245, -2.3005548, 2.2798166
5: -8.3676090, -5.7517304, -8.3798866, -5.7470345, -2.0501947, 2.0562913
6: -12.7108335, -9.7072506, -12.9543848, -9.7024498, -2.3107891, 2.3490751
7: -6.2174664, -3.3342378, -6.2408166, -3.3283572, -2.7980561, 2.8086867
8: -3.0022531, -0.2282829, -3.0490673, -0.2229137, -2.2729635, 2.3259883
9: -5.4689026, -3.2161713, -5.4763145, -3.1923089, -1.7068019, 1.7260952

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1248859, upper bound: 1.1096738
time: 5.56 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255589, upper bound: 1.1255559
time: 7.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.38 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1136499, upper bound: 1.0951101
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1136499, upper bound: 1.1057439
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1136500, upper bound: 1.1003082
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1136500, upper bound: 1.1109068
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1242308, upper bound: 1.0951126
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1242320, upper bound: 1.0953926
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1242308, upper bound: 1.1003102
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1242320, upper bound: 1.1003074
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1248378, upper bound: 1.0938138
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1255109, upper bound: 1.1094723
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1248378, upper bound: 1.0990094
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1255109, upper bound: 1.1147221
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1248859, upper bound: 1.1045073
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1255589, upper bound: 1.1203612
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1248859, upper bound: 1.1096738
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 28.38
Output dim: 4, lower bound: -1.1255589, upper bound: 1.1255559

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2286644, -11.1461411, -14.2562952, -11.1342220, -2.4938602, 2.5097930
1: -10.5773792, -7.9478159, -10.6018066, -7.9216909, -2.0290537, 2.0563219
2: -10.0836191, -7.4035726, -10.1197329, -7.3596563, -2.2985735, 2.2975783
3: -12.7258101, -10.3885622, -12.7564878, -10.3651943, -1.9544082, 1.9552860
4: 5.9702740, 8.3526268, 5.9205532, 8.3947048, -2.1223679, 2.1895771
5: -8.3286247, -5.7994957, -8.3523064, -5.7710247, -1.9743052, 1.9763496
6: -12.6807117, -9.8238678, -12.7030287, -9.7990618, -2.1114359, 2.1996622
7: -6.1278739, -3.3907146, -6.1808395, -3.3525748, -2.6840935, 2.5578318
8: -2.9503045, -0.2853184, -2.9786010, -0.2510328, -2.2068229, 2.2119079
9: -5.3653774, -3.2671280, -5.4187737, -3.2264495, -1.6255741, 1.5001087

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012779, upper bound: 1.0951101
time: 5.18 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012780, upper bound: 1.0951097
time: 6.25 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2286644, -11.1461411, -14.4090281, -11.0706425, -2.5561919, 2.5482199
1: -10.5773792, -7.9478159, -10.6130152, -7.9082761, -2.0424407, 2.0672367
2: -10.0836191, -7.4035726, -10.1829128, -7.3336544, -2.3248773, 2.3399200
3: -12.7258101, -10.3885622, -12.7713270, -10.3544540, -1.9646473, 1.9709966
4: 5.9702740, 8.3526268, 5.8850365, 8.4500847, -2.1296134, 2.2258234
5: -8.3286247, -5.7994957, -8.3697395, -5.7618570, -1.9827552, 1.9941413
6: -12.6807117, -9.8238678, -12.9488297, -9.7207222, -2.1897788, 2.2347460
7: -6.1278739, -3.3907146, -6.2105336, -3.3453386, -2.6915522, 2.5867071
8: -2.9503045, -0.2853184, -3.0281546, -0.2318716, -2.2260141, 2.2578230
9: -5.3653774, -3.2671280, -5.4325147, -3.2005148, -1.6502924, 1.5103168

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012779, upper bound: 1.1057440
time: 5.04 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012780, upper bound: 1.1057465
time: 5.94 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -14.2385397, -11.1378145, -14.2583008, -11.1325150, -2.5109215, 2.5198410
1: -10.5824404, -7.9380198, -10.6033802, -7.9200602, -2.0355878, 2.0689268
2: -10.1214342, -7.3820987, -10.1361332, -7.3592310, -2.3055835, 2.3384819
3: -12.7448978, -10.3747292, -12.7639761, -10.3640995, -1.9625874, 1.9765403
4: 5.9440761, 8.3738384, 5.9173651, 8.4049778, -2.1506972, 2.1940193
5: -8.3340034, -5.7874694, -8.3531046, -5.7692547, -1.9834213, 1.9889333
6: -12.6915693, -9.8183794, -12.7042542, -9.7974119, -2.1241879, 2.2045336
7: -6.1476908, -3.3673999, -6.1830988, -3.3423083, -2.7140074, 2.5623126
8: -2.9832954, -0.2614679, -2.9935050, -0.2501655, -2.2131672, 2.2508941
9: -5.3876624, -3.2515154, -5.4293199, -3.2258492, -1.6245117, 1.5185037

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012779, upper bound: 1.1003076
time: 6.74 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012779, upper bound: 1.1003078
time: 8.04 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -14.2385397, -11.1378145, -14.4110432, -11.0689545, -2.5732470, 2.5547969
1: -10.5824404, -7.9380198, -10.6145868, -7.9066324, -2.0489788, 2.0798440
2: -10.1214342, -7.3820987, -10.1992941, -7.3332257, -2.3318849, 2.3652661
3: -12.7448978, -10.3747292, -12.7787971, -10.3533325, -1.9728341, 1.9922342
4: 5.9440761, 8.3738384, 5.8819246, 8.4603567, -2.1575532, 2.2303567
5: -8.3340034, -5.7874694, -8.3705206, -5.7600899, -1.9918737, 2.0067437
6: -12.6915693, -9.8183794, -12.9500589, -9.7190800, -2.2025189, 2.2374630
7: -6.1476908, -3.3673999, -6.2127314, -3.3350849, -2.7214537, 2.5912228
8: -2.9832954, -0.2614679, -3.0430305, -0.2310004, -2.2323470, 2.2833898
9: -5.3876624, -3.2515154, -5.4430542, -3.1999025, -1.6480970, 1.5287383

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012779, upper bound: 1.1109067
time: 5.44 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1012780, upper bound: 1.1109066
time: 5.29 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -14.3813486, -11.0826035, -14.2562952, -11.1342220, -2.5317440, 2.5721054
1: -10.5886211, -7.9343839, -10.6018066, -7.9216909, -2.0399680, 2.0698054
2: -10.1468029, -7.3777137, -10.1197329, -7.3596563, -2.3350029, 2.3238142
3: -12.7407513, -10.3777542, -12.7564878, -10.3651943, -1.9698286, 1.9655886
4: 5.9349623, 8.4080296, 5.9205532, 8.3947048, -2.1528587, 2.2107072
5: -8.3459930, -5.7902865, -8.3523064, -5.7710247, -1.9922767, 1.9848461
6: -12.9264708, -9.7455387, -12.7030287, -9.7990618, -2.1535397, 2.2780132
7: -6.1575851, -3.3834004, -6.1808395, -3.3525748, -2.7129955, 2.5653086
8: -2.9998751, -0.2661953, -2.9786010, -0.2510328, -2.2565103, 2.2310171
9: -5.3791370, -3.2411623, -5.4187737, -3.2264495, -1.6374340, 1.5090032

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1118750, upper bound: 1.0951096
time: 4.76 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1118750, upper bound: 1.0951126
time: 5.85 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -14.3813486, -11.0826035, -14.4090309, -11.0706444, -2.5638947, 2.5798903
1: -10.5886211, -7.9343839, -10.6130152, -7.9082732, -2.0539641, 2.0813296
2: -10.1468029, -7.3777137, -10.1829128, -7.3336420, -2.3471889, 2.3461521
3: -12.7407513, -10.3777542, -12.7713270, -10.3544397, -1.9752851, 1.9765043
4: 5.9349623, 8.4080296, 5.8850360, 8.4500847, -2.1610885, 2.2436013
5: -8.3459930, -5.7902865, -8.3697414, -5.7618570, -2.0052800, 2.0084057
6: -12.9264708, -9.7455387, -12.9488297, -9.7207222, -2.1878819, 2.2762480
7: -6.1575851, -3.3834004, -6.2105370, -3.3453329, -2.7210999, 2.5948248
8: -2.9998751, -0.2661953, -3.0281644, -0.2318721, -2.2466459, 2.2517090
9: -5.3791370, -3.2411623, -5.4325142, -3.2005134, -1.6466651, 1.5193377

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1118761, upper bound: 1.0951096
time: 5.78 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1118762, upper bound: 1.0951099
time: 7.06 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -14.3912430, -11.0742779, -14.2583008, -11.1325150, -2.5423379, 2.5821290
1: -10.5936651, -7.9245815, -10.6033802, -7.9200602, -2.0464916, 2.0824146
2: -10.1845932, -7.3561811, -10.1361332, -7.3592310, -2.3384819, 2.3647282
3: -12.7597628, -10.3639030, -12.7639761, -10.3640995, -1.9779282, 1.9868491
4: 5.9087849, 8.4292326, 5.9173651, 8.4049778, -2.1811013, 2.2127233
5: -8.3513060, -5.7782669, -8.3531046, -5.7692547, -2.0006194, 1.9974327
6: -12.9373627, -9.7400494, -12.7042542, -9.7974119, -2.1620698, 2.2828698
7: -6.1772947, -3.3601322, -6.1830988, -3.3423083, -2.7430153, 2.5697622
8: -3.0328188, -0.2423525, -2.9935050, -0.2501655, -2.2600570, 2.2699943
9: -5.4014378, -3.2255387, -5.4293199, -3.2258492, -1.6363063, 1.5274038

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1118750, upper bound: 1.1003072
time: 5.11 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1118750, upper bound: 1.1003072
time: 5.11 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -14.3912430, -11.0742779, -14.4110460, -11.0689583, -2.5809927, 2.5898728
1: -10.5936651, -7.9245815, -10.6145887, -7.9066310, -2.0604901, 2.0939393
2: -10.1845932, -7.3561811, -10.1992970, -7.3332157, -2.3542929, 2.3870358
3: -12.7597628, -10.3639030, -12.7787991, -10.3533173, -1.9833941, 1.9977474
4: 5.9087849, 8.4292326, 5.8819246, 8.4603577, -2.1893253, 2.2457309
5: -8.3513060, -5.7782669, -8.3705225, -5.7600908, -2.0110831, 2.0210061
6: -12.9373627, -9.7400494, -12.9500608, -9.7190781, -2.2006602, 2.2811103
7: -6.1772947, -3.3601322, -6.2127342, -3.3350792, -2.7511082, 2.5993128
8: -3.0328188, -0.2423525, -3.0430408, -0.2310004, -2.2529411, 2.2906566
9: -5.4014378, -3.2255387, -5.4430542, -3.1999035, -1.6455498, 1.5377619

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.308117389678955
rel_dist={4: [-1.1255949748543852, 1.1255974909157898]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982736, upper bound: 1.0081493
time: 5.09 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090097, upper bound: 1.0090106
time: 6.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.44 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 4, lower bound: -0.9982736, upper bound: 1.0081493
IS_B2, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 4, lower bound: -1.0090097, upper bound: 1.0090106

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -14.2564106, -11.0838490, -14.2392712, -11.0884342, -2.4459372, 2.4365983
1: -10.6055689, -7.9192228, -10.5863619, -7.9344053, -1.9506738, 1.9495702
2: -10.1383352, -7.3426933, -10.1246023, -7.3621345, -2.2606950, 2.2695012
3: -12.7662840, -10.3635712, -12.7497921, -10.3731194, -1.9064336, 1.8977566
4: 5.8982973, 8.4014158, 5.9232817, 8.3751202, -2.1198206, 2.1141953
5: -8.3564529, -5.7673407, -8.3391552, -5.7829981, -1.9162126, 1.9117804
6: -12.7058392, -9.7270794, -12.6938314, -9.7448015, -2.0902939, 2.0775115
7: -6.1839523, -3.3422689, -6.1540017, -3.3660154, -2.5171375, 2.5102921
8: -2.9951921, -0.2378831, -2.9860821, -0.2476349, -2.1831264, 2.1841512
9: -5.4291468, -3.2247992, -5.3939929, -3.2494354, -1.4686584, 1.4638321

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906212, upper bound: 1.0081376
time: 10.25 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982685, upper bound: 1.0081430
time: 6.79 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -14.2722263, -11.0764141, -14.2722225, -11.0764179, -2.4655304, 2.4934635
1: -10.6166248, -7.9022160, -10.6166220, -7.9022179, -2.0254946, 2.0394907
2: -10.1443129, -7.3213792, -10.1443119, -7.3213830, -2.3280787, 2.3304877
3: -12.7821169, -10.3563175, -12.7821121, -10.3563175, -1.9555011, 1.9442852
4: 5.8858538, 8.4309311, 5.8858571, 8.4309235, -2.2282734, 2.2495747
5: -8.3676186, -5.7517157, -8.3676128, -5.7517204, -1.9603539, 1.9654820
6: -12.7108421, -9.7072105, -12.7108374, -9.7072144, -2.2080922, 2.2138486
7: -6.2174859, -3.3342147, -6.2174792, -3.3342180, -2.7381716, 2.7234015
8: -3.0022955, -0.2282639, -3.0022945, -0.2282662, -2.2263441, 2.2275944
9: -5.4689360, -3.2161660, -5.4689250, -3.2161689, -1.6696615, 1.6465502

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013777, upper bound: 1.0090071
time: 6.23 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090050, upper bound: 1.0090063
time: 5.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.16 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 4, lower bound: -0.9906212, upper bound: 1.0081376
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 4, lower bound: -0.9982685, upper bound: 1.0081430
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 4, lower bound: -1.0013777, upper bound: 1.0090071
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 27.16
Output dim: 4, lower bound: -1.0090050, upper bound: 1.0090063

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9867606, upper bound: 1.0081271
time: 4.81 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0081301
time: 4.49 seconds

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9944043, upper bound: 1.0081359
time: 5.73 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982613, upper bound: 1.0081333
time: 9.26 seconds

## BFS IS instance: IS_B2_A1

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

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9975051, upper bound: 1.0089966
time: 4.84 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013681, upper bound: 1.0090000
time: 4.60 seconds

## BFS IS instance: IS_B2_A2

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

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0051273, upper bound: 1.0089965
time: 4.38 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089954, upper bound: 1.0089981
time: 5.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.68 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -0.9867606, upper bound: 1.0081271
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0081301
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -0.9944043, upper bound: 1.0081359
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -0.9982613, upper bound: 1.0081333
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -0.9975051, upper bound: 1.0089966
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -1.0013681, upper bound: 1.0090000
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -1.0051273, upper bound: 1.0089965
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.68
Output dim: 4, lower bound: -1.0089954, upper bound: 1.0089981

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.2532997, -11.1352463, -14.2290955, -11.1165438, -2.4065008, 2.3665152
1: -10.5997810, -7.9247608, -10.5797386, -7.9456468, -1.9327974, 1.9372227
2: -10.1157866, -7.3631916, -10.0855160, -7.3916216, -2.1991415, 2.2049735
3: -12.7525549, -10.3664379, -12.7287350, -10.3876333, -1.8784480, 1.8738899
4: 5.9228392, 8.3880024, 5.9578705, 8.3533859, -2.0733223, 2.0658579
5: -8.3504009, -5.7739048, -8.3317566, -5.7968249, -1.8929038, 1.8921907
6: -12.7021236, -9.8025999, -12.6820698, -9.7797022, -2.0522561, 1.9901023
7: -6.1749830, -3.3557715, -6.1316357, -3.3898988, -2.4833727, 2.4734187
8: -2.9748120, -0.2527485, -2.9519501, -0.2770290, -2.1335497, 2.1355119
9: -5.4103565, -3.2275867, -5.3691730, -3.2658882, -1.4294548, 1.4364338

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9867609, upper bound: 1.0005044
time: 6.93 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9867606, upper bound: 1.0081271
time: 4.68 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.2556686, -11.1332340, -14.2389717, -11.1081905, -2.4164929, 2.3835878
1: -10.6016407, -7.9228334, -10.5847940, -7.9358568, -1.9456728, 1.9440343
2: -10.1351633, -7.3626909, -10.1233196, -7.3701315, -2.2433491, 2.2101417
3: -12.7614002, -10.3651419, -12.7478209, -10.3737774, -1.9009819, 1.8818583
4: 5.9190788, 8.4001417, 5.9316139, 8.3745956, -2.0771046, 2.0976880
5: -8.3513489, -5.7718134, -8.3371029, -5.7847967, -1.9053197, 1.9015360
6: -12.7035713, -9.8006506, -12.6929245, -9.7742252, -2.0572519, 2.0032036
7: -6.1776552, -3.3436356, -6.1514606, -3.3665872, -2.4871812, 2.5052061
8: -2.9924254, -0.2517238, -2.9849429, -0.2531710, -2.1753078, 2.1406279
9: -5.4228201, -3.2268782, -5.3914490, -3.2502661, -1.4492939, 1.4342058

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0005073
time: 4.74 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0081301
time: 4.50 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.4060307, -11.0716629, -14.2293816, -11.0968199, -2.4648948, 2.4213400
1: -10.6109867, -7.9113421, -10.5813007, -7.9442034, -1.9456921, 1.9522128
2: -10.1789703, -7.3371954, -10.0867710, -7.3836594, -2.2516255, 2.2343891
3: -12.7673988, -10.3556919, -12.7306852, -10.3870010, -1.8941197, 1.8860903
4: 5.8873301, 8.4433823, 5.9496078, 8.3538876, -2.1040025, 2.0847077
5: -8.3678045, -5.7647381, -8.3338242, -5.7950382, -1.9173841, 1.9035375
6: -12.9479198, -9.7242565, -12.6829739, -9.7503109, -2.1186714, 2.0368881
7: -6.2046919, -3.3485231, -6.1341572, -3.3893461, -2.5127878, 2.4835548
8: -3.0243812, -0.2335930, -2.9530473, -0.2715144, -2.1853118, 2.1518173
9: -5.4241023, -3.2016506, -5.3716974, -3.2650657, -1.4398053, 1.4632185

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9943927, upper bound: 1.0005032
time: 6.34 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9943938, upper bound: 1.0005038
time: 5.35 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.4084110, -11.0696812, -14.2392664, -11.0884552, -2.4710684, 2.4384542
1: -10.6128407, -7.9094005, -10.5863495, -7.9344139, -1.9585636, 1.9590256
2: -10.1983242, -7.3366895, -10.1245670, -7.3621583, -2.2792583, 2.2395916
3: -12.7762280, -10.3543816, -12.7497702, -10.3731232, -1.9166451, 1.8940668
4: 5.8836575, 8.4555244, 5.9233074, 8.3750963, -2.1079197, 2.1140428
5: -8.3687315, -5.7626452, -8.3391485, -5.7830091, -1.9298034, 1.9128907
6: -12.9493809, -9.7223158, -12.6938295, -9.7448406, -2.1210613, 2.0499747
7: -6.2072916, -3.3364019, -6.1539869, -3.3660364, -2.5166464, 2.5153303
8: -3.0419612, -0.2325649, -2.9860411, -0.2476530, -2.2130013, 2.1569242
9: -5.4365602, -3.2009268, -5.3939695, -3.2494388, -1.4596653, 1.4597807

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982588, upper bound: 0.9997051
time: 8.09 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982613, upper bound: 1.0081333
time: 9.20 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.2690678, -11.1277876, -14.2619028, -11.1044979, -2.4260931, 2.4232130
1: -10.6108255, -7.9077477, -10.6100683, -7.9134583, -2.0075877, 2.0271845
2: -10.1217527, -7.3419185, -10.1052322, -7.3510075, -2.2664199, 2.2659149
3: -12.7683830, -10.3592186, -12.7610893, -10.3708916, -1.9273562, 1.9206514
4: 5.9104557, 8.4175062, 5.9206271, 8.4091930, -2.1815066, 2.1956158
5: -8.3615932, -5.7582865, -8.3602686, -5.7655230, -1.9370818, 1.9458485
6: -12.7071323, -9.7827063, -12.6990566, -9.7420712, -2.1701140, 2.1263771
7: -6.2085352, -3.3477018, -6.1951113, -3.3580461, -2.7044201, 2.6865025
8: -2.9819198, -0.2431397, -2.9681911, -0.2577047, -2.1767282, 2.1789246
9: -5.4501562, -3.2189598, -5.4441233, -3.2326460, -1.6304638, 1.6191804

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9859331, upper bound: 1.0085501
time: 4.56 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974995, upper bound: 1.0089918
time: 4.73 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.2714748, -11.1257524, -14.2719154, -11.0961571, -2.4361181, 2.4405112
1: -10.6126747, -7.9058199, -10.6150408, -7.9036670, -2.0204601, 2.0339127
2: -10.1411295, -7.3414049, -10.1430159, -7.3294039, -2.3107352, 2.2711165
3: -12.7772350, -10.3578911, -12.7801418, -10.3569584, -1.9500794, 1.9284024
4: 5.9066420, 8.4296513, 5.8941836, 8.4303961, -2.1855540, 2.2253661
5: -8.3625011, -5.7561922, -8.3655739, -5.7535229, -1.9494901, 1.9552627
6: -12.7085724, -9.7807713, -12.7099295, -9.7366333, -2.1751328, 2.1395657
7: -6.2112064, -3.3355756, -6.2149496, -3.3347826, -2.7082496, 2.7183270
8: -2.9995270, -0.2421103, -3.0011559, -0.2338095, -2.2184772, 2.1840587
9: -5.4626093, -3.2182469, -5.4663811, -3.2169986, -1.6502790, 1.6169262

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9897818, upper bound: 1.0085508
time: 7.19 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013624, upper bound: 1.0089919
time: 4.03 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.4218311, -11.0641956, -14.2622013, -11.0847874, -2.4878893, 2.4780393
1: -10.6220541, -7.8943481, -10.6116409, -7.9120188, -2.0205426, 2.0421233
2: -10.1849585, -7.3158188, -10.1064997, -7.3430195, -2.3158221, 2.2952354
3: -12.7832088, -10.3484583, -12.7630396, -10.3702650, -1.9426031, 1.9328046
4: 5.8748217, 8.4728737, 5.9123626, 8.4096975, -2.2133112, 2.2119074
5: -8.3789806, -5.7491274, -8.3623228, -5.7637315, -1.9614892, 1.9571488
6: -12.9529381, -9.7043810, -12.6999607, -9.7126884, -2.2318702, 2.1732216
7: -6.2382140, -3.3404722, -6.1976204, -3.3575003, -2.7339296, 2.6966200
8: -3.0314877, -0.2239475, -2.9692879, -0.2521858, -2.2267909, 2.1952157
9: -5.4638615, -3.1930356, -5.4466505, -3.2318254, -1.6407061, 1.6408339

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9935683, upper bound: 1.0085599
time: 5.68 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0051216, upper bound: 1.0089915
time: 4.19 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.4242458, -11.0621967, -14.2722168, -11.0764370, -2.4941058, 2.4953756
1: -10.6239023, -7.8924065, -10.6166124, -7.9022264, -2.0334110, 2.0488963
2: -10.2043104, -7.3153000, -10.1442757, -7.3214064, -2.3434634, 2.3004637
3: -12.7920437, -10.3471203, -12.7820873, -10.3563232, -1.9653163, 1.9405637
4: 5.8710995, 8.4850216, 5.8858814, 8.4308987, -2.2174664, 2.2416592
5: -8.3798857, -5.7470369, -8.3676090, -5.7517314, -1.9706063, 1.9665706
6: -12.9543858, -9.7024517, -12.7108326, -9.7072554, -2.2342901, 2.1863940
7: -6.2408142, -3.3283610, -6.2174654, -3.3342378, -2.7377872, 2.7284322
8: -3.0490632, -0.2229137, -3.0022526, -0.2282853, -2.2544231, 2.2003388
9: -5.4763098, -3.1923113, -5.4689012, -3.2161727, -1.6605422, 1.6373699

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974228, upper bound: 1.0085559
time: 4.84 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089898, upper bound: 1.0089950
time: 5.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.80 seconds
IS_B1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9867609, upper bound: 1.0005044
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9867606, upper bound: 1.0081271
IS_B1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0005073
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0081301
IS_B1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9943927, upper bound: 1.0005032
IS_B1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9943938, upper bound: 1.0005038
IS_B1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9982588, upper bound: 0.9997051
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9982613, upper bound: 1.0081333
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9859331, upper bound: 1.0085501
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9974995, upper bound: 1.0089918
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9897818, upper bound: 1.0085508
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -1.0013624, upper bound: 1.0089919
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9935683, upper bound: 1.0085599
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -1.0051216, upper bound: 1.0089915
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -0.9974228, upper bound: 1.0085559
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.80
Output dim: 4, lower bound: -1.0089898, upper bound: 1.0089950

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -14.2532997, -11.1352463, -14.3813457, -11.0826073, -2.4373455, 2.4006355
1: -10.5997810, -7.9247608, -10.5886202, -7.9343848, -1.9439270, 1.9457409
2: -10.1157866, -7.3631916, -10.1468039, -7.3777246, -2.2116189, 2.2360756
3: -12.7525549, -10.3664379, -12.7407513, -10.3777695, -1.8878779, 1.8868146
4: 5.9228392, 8.3880024, 5.9349637, 8.4080296, -2.0947409, 2.0840788
5: -8.3504009, -5.7739048, -8.3459892, -5.7902875, -1.8979988, 1.9057884
6: -12.7021236, -9.8025999, -12.9264660, -9.7455368, -2.0827174, 2.0341377
7: -6.1749830, -3.3557715, -6.1575813, -3.3834059, -2.4894023, 2.4982195
8: -2.9748120, -0.2527485, -2.9998651, -0.2661963, -2.1444030, 2.1838479
9: -5.4103565, -3.2275867, -5.3791361, -3.2411642, -1.4369476, 1.4444954

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9867607, upper bound: 0.9996934
time: 4.85 seconds

## Relational analysis of IS_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9867634, upper bound: 1.0081301
time: 5.50 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -14.2556686, -11.1332340, -14.3912392, -11.0742760, -2.4472842, 2.4112391
1: -10.6016407, -7.9228334, -10.5936651, -7.9245834, -1.9568131, 1.9525483
2: -10.1351633, -7.3626909, -10.1845942, -7.3561926, -2.2558351, 2.2375181
3: -12.7614002, -10.3651419, -12.7597609, -10.3639174, -1.9104037, 1.8946991
4: 5.9190788, 8.4001417, 5.9087858, 8.4292326, -2.0955882, 2.1136801
5: -8.3513489, -5.7718134, -8.3513031, -5.7782655, -1.9104176, 1.9145064
6: -12.7035713, -9.8006506, -12.9373608, -9.7400494, -2.0850997, 2.0429921
7: -6.1776552, -3.3436356, -6.1772919, -3.3601384, -2.4931884, 2.5301108
8: -2.9924254, -0.2517238, -3.0328083, -0.2423525, -2.1861429, 2.1844351
9: -5.4228201, -3.2268782, -5.4014368, -3.2255406, -1.4567952, 1.4422045

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of IS_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906116, upper bound: 0.9996929
time: 6.59 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0081301
time: 4.54 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -14.4240236, -11.0627098, -14.2392664, -11.0884552, -2.4759417, 2.4398341
1: -10.6225643, -7.8926411, -10.5863495, -7.9344139, -1.9659569, 1.9767842
2: -10.2039433, -7.3154340, -10.1245670, -7.3621583, -2.2777457, 2.2582583
3: -12.7918558, -10.3488188, -12.7497702, -10.3731232, -1.9325404, 1.8980341
4: 5.8750963, 8.4849691, 5.9233074, 8.3750963, -2.1114283, 2.1205254
5: -8.3790359, -5.7471571, -8.3391485, -5.7830091, -1.9330111, 1.9197667
6: -12.9526625, -9.7025309, -12.6938295, -9.7448406, -2.1114798, 2.0827551
7: -6.2401953, -3.3286474, -6.1539869, -3.3660364, -2.5271835, 2.5175323
8: -3.0478270, -0.2230778, -2.9860411, -0.2476530, -2.2171488, 2.1696239
9: -5.4759874, -3.1971159, -5.3939695, -3.2494388, -1.4639096, 1.4451849

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982433, upper bound: 1.0005071
time: 5.55 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982447, upper bound: 1.0005033
time: 5.64 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -14.2669373, -11.1292381, -14.2566929, -11.1064720, -2.4217024, 2.4168015
1: -10.6087666, -7.9109964, -10.6026421, -7.9193878, -1.9994869, 2.0138869
2: -10.1208878, -7.3466654, -10.0997353, -7.3598347, -2.2554359, 2.2526586
3: -12.7645063, -10.3607121, -12.7533436, -10.3755131, -1.9178743, 1.9112988
4: 5.9143105, 8.4096689, 5.9402871, 8.3957767, -2.1608090, 2.1639943
5: -8.3592882, -5.7606926, -8.3534946, -5.7702999, -1.9298735, 1.9357138
6: -12.7060013, -9.7904444, -12.6881170, -9.7553291, -2.1547070, 2.1073270
7: -6.1978407, -3.3488755, -6.1768236, -3.3722200, -2.6785994, 2.6667218
8: -2.9803238, -0.2473040, -2.9611883, -0.2654538, -2.1662035, 2.1672206
9: -5.4383984, -3.2217684, -5.4238787, -3.2522273, -1.5969341, 1.5973651

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9859328, upper bound: 1.0009024
time: 5.27 seconds

## Relational analysis of IS_B2_A1_B1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9859328, upper bound: 1.0085501
time: 5.21 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -14.2685261, -11.1277885, -14.2609749, -11.1044989, -2.4254713, 2.4248476
1: -10.6108208, -7.9077659, -10.6100674, -7.9134898, -2.0066514, 2.0296261
2: -10.1217527, -7.3419342, -10.1052313, -7.3510323, -2.2657347, 2.2676845
3: -12.7683249, -10.3592205, -12.7609882, -10.3708944, -1.9286399, 1.9200749
4: 5.9104614, 8.4175024, 5.9206362, 8.4091854, -2.1760526, 2.1857734
5: -8.3615913, -5.7583170, -8.3602638, -5.7655745, -1.9368157, 1.9464352
6: -12.7070923, -9.7827101, -12.6989813, -9.7420788, -2.1691852, 2.1263287
7: -6.2085304, -3.3477268, -6.1951041, -3.3580890, -2.7004557, 2.6863179
8: -2.9819007, -0.2431436, -2.9681597, -0.2577114, -2.1765456, 2.1788993
9: -5.4501481, -3.2189617, -5.4441123, -3.2326488, -1.6173546, 1.6128616

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 5735

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9974996, upper bound: 1.0013640
time: 4.76 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974995, upper bound: 1.0089920
time: 5.12 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -14.2693367, -11.1272078, -14.2667017, -11.0981388, -2.4316988, 2.4340692
1: -10.6106148, -7.9090695, -10.6076288, -7.9095955, -2.0123634, 2.0206251
2: -10.1402645, -7.3461523, -10.1375265, -7.3382611, -2.2997422, 2.2578599
3: -12.7733593, -10.3593874, -12.7724133, -10.3615875, -1.9405847, 1.9190655
4: 5.9105105, 8.4218140, 5.9139280, 8.4169846, -2.1648493, 2.1936760
5: -8.3602028, -5.7586007, -8.3588371, -5.7582998, -1.9422770, 1.9451160
6: -12.7074432, -9.7885075, -12.6989784, -9.7498875, -2.1597219, 2.1205039
7: -6.2005095, -3.3367491, -6.1966605, -3.3489499, -2.6824374, 2.6985259
8: -2.9979329, -0.2462749, -2.9941626, -0.2415590, -2.2079530, 2.1723642
9: -5.4508548, -3.2210550, -5.4461355, -3.2365956, -1.6167526, 1.5951188

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9897818, upper bound: 1.0009031
time: 6.56 seconds

## Relational analysis of IS_B2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9897818, upper bound: 1.0085508
time: 7.02 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -14.2709408, -11.1257591, -14.2709980, -11.0961609, -2.4354954, 2.4421473
1: -10.6126709, -7.9058399, -10.6150389, -7.9036984, -2.0195231, 2.0363560
2: -10.1411285, -7.3414211, -10.1430130, -7.3294296, -2.3100500, 2.2728853
3: -12.7771749, -10.3578920, -12.7800398, -10.3569584, -1.9513636, 1.9278250
4: 5.9066486, 8.4296474, 5.8941946, 8.4303913, -2.1800995, 2.2155037
5: -8.3624992, -5.7562242, -8.3655682, -5.7535753, -1.9492235, 1.9540629
6: -12.7085304, -9.7807751, -12.7098579, -9.7366409, -2.1742029, 2.1395164
7: -6.2112007, -3.3355997, -6.2149420, -3.3348246, -2.7016711, 2.7138371
8: -2.9995093, -0.2421141, -3.0011244, -0.2338138, -2.2182932, 2.1840315
9: -5.4626026, -3.2182479, -5.4663677, -3.2170029, -1.6371689, 1.6106081

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 5735

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013624, upper bound: 1.0013639
time: 4.15 seconds

## Relational analysis of IS_B2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013624, upper bound: 1.0089919
time: 3.95 seconds

## BFS IS instance: IS_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -14.4196930, -11.0656452, -14.2569885, -11.0867586, -2.4792585, 2.4716282
1: -10.6199932, -7.8975883, -10.6042128, -7.9179440, -2.0124412, 2.0288196
2: -10.1840878, -7.3205934, -10.1010017, -7.3518558, -2.3042731, 2.2820046
3: -12.7793360, -10.3499374, -12.7552919, -10.3748808, -1.9331274, 1.9234648
4: 5.8787079, 8.4650431, 5.9320254, 8.3962784, -2.1925611, 2.1802721
5: -8.3767014, -5.7515330, -8.3555565, -5.7685089, -1.9543114, 1.9470284
6: -12.9518042, -9.7121181, -12.6890221, -9.7259464, -2.2163277, 2.1541748
7: -6.2275267, -3.3416455, -6.1793365, -3.3716729, -2.7081079, 2.6768398
8: -3.0298946, -0.2281208, -2.9622850, -0.2599354, -2.2161827, 2.1835108
9: -5.4521174, -3.1958423, -5.4264040, -3.2514052, -1.6071911, 1.6188321

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9935625, upper bound: 1.0009024
time: 6.70 seconds

## Relational analysis of IS_B2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9935636, upper bound: 1.0009350
time: 5.51 seconds

## BFS IS instance: IS_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -14.4212952, -11.0641966, -14.2612705, -11.0847931, -2.4864964, 2.4796753
1: -10.6220541, -7.8943648, -10.6116400, -7.9120493, -2.0196064, 2.0445657
2: -10.1849566, -7.3158340, -10.1064987, -7.3430452, -2.3127561, 2.2970047
3: -12.7831516, -10.3484573, -12.7629366, -10.3702669, -1.9438891, 1.9322283
4: 5.8748283, 8.4728699, 5.9123726, 8.4096899, -2.2078562, 2.2020683
5: -8.3789787, -5.7491579, -8.3623190, -5.7637858, -1.9611287, 1.9577374
6: -12.9528971, -9.7043848, -12.6998863, -9.7126942, -2.2262106, 2.1731730
7: -6.2382088, -3.3404958, -6.1976128, -3.3575416, -2.7297544, 2.6964364
8: -3.0314686, -0.2239494, -2.9692566, -0.2521906, -2.2243590, 2.1951904
9: -5.4638548, -3.1930380, -5.4466362, -3.2318277, -1.6275971, 1.6274228

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 5735

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A2_B1_B2_B1

### Relational analysis result of IS_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0051211, upper bound: 1.0013632
time: 4.40 seconds

## Relational analysis of IS_B2_A2_B1_B2_B2

### Relational analysis result of IS_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0051222, upper bound: 1.0013633
time: 4.32 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -14.4221020, -11.0636454, -14.2669992, -11.0784168, -2.4854698, 2.4889345
1: -10.6218443, -7.8956451, -10.6091948, -7.9081531, -2.0253119, 2.0356016
2: -10.2034407, -7.3200788, -10.1387863, -7.3302679, -2.3319159, 2.2872262
3: -12.7881708, -10.3486042, -12.7743587, -10.3609457, -1.9558268, 1.9312396
4: 5.8749967, 8.4771872, 5.9056301, 8.4174843, -2.1967106, 2.2099681
5: -8.3776112, -5.7494421, -8.3608780, -5.7565084, -1.9633651, 1.9564381
6: -12.9532528, -9.7101860, -12.6998806, -9.7205105, -2.2187471, 2.1673336
7: -6.2301273, -3.3295336, -6.1991787, -3.3484054, -2.7119746, 2.7086315
8: -3.0474710, -0.2270880, -2.9952593, -0.2360382, -2.2438183, 2.1886449
9: -5.4645700, -3.1951170, -5.4486556, -3.2357678, -1.6270318, 1.6153717

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9897819, upper bound: 0.9974174
time: 4.65 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974228, upper bound: 1.0085598
time: 5.90 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -14.4237156, -11.0621967, -14.2713013, -11.0764389, -2.4927030, 2.4970121
1: -10.6238995, -7.8924232, -10.6166096, -7.9022584, -2.0324750, 2.0513358
2: -10.2043104, -7.3153172, -10.1442719, -7.3214331, -2.3403964, 2.3022325
3: -12.7919865, -10.3471203, -12.7819872, -10.3563232, -1.9666014, 1.9399881
4: 5.8711042, 8.4850149, 5.8858943, 8.4308920, -2.2120118, 2.2317991
5: -8.3798809, -5.7470665, -8.3676033, -5.7517838, -1.9689360, 1.9654343
6: -12.9543438, -9.7024555, -12.7107601, -9.7072601, -2.2286339, 2.1863456
7: -6.2408094, -3.3283839, -6.2174568, -3.3342788, -2.7310491, 2.7239881
8: -3.0490434, -0.2229166, -3.0022211, -0.2282906, -2.2519922, 2.2003136
9: -5.4763012, -3.1923118, -5.4688888, -3.2161746, -1.6474328, 1.6239603

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 5735

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of IS_B2_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089893, upper bound: 1.0013669
time: 5.09 seconds

## Relational analysis of IS_B2_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089904, upper bound: 1.0013636
time: 5.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.29 seconds
IS_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9867607, upper bound: 0.9996934
IS_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9867634, upper bound: 1.0081301
IS_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9906116, upper bound: 0.9996929
IS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9906116, upper bound: 1.0081301
IS_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9982433, upper bound: 1.0005071
IS_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9982447, upper bound: 1.0005033
IS_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9859328, upper bound: 1.0009024
IS_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9859328, upper bound: 1.0085501
IS_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9974996, upper bound: 1.0013640
IS_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9974995, upper bound: 1.0089920
IS_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9897818, upper bound: 1.0009031
IS_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9897818, upper bound: 1.0085508
IS_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -1.0013624, upper bound: 1.0013639
IS_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -1.0013624, upper bound: 1.0089919
IS_B2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9935625, upper bound: 1.0009024
IS_B2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9935636, upper bound: 1.0009350
IS_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -1.0051211, upper bound: 1.0013632
IS_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -1.0051222, upper bound: 1.0013633
IS_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9897819, upper bound: 0.9974174
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -0.9974228, upper bound: 1.0085598
IS_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -1.0089893, upper bound: 1.0013669
IS_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.29
Output dim: 4, lower bound: -1.0089904, upper bound: 1.0013636
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.48 seconds
