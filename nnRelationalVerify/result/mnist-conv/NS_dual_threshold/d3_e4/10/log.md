## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0070000638


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4699616, 2.4699612)
1: (-10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0266666, 2.0266664)
2: (-10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3282871, 2.3282874)
3: (-12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9452214, 1.9452219)
4: (5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2497125, 2.2497125)
5: (-8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9607882, 1.9607880)
6: (-12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2138529, 2.2138529)
7: (-6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7246361, 2.7246356)
8: (-3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2275991, 2.2275991)
9: (-5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6726398, 1.6726398)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.12 + 34.28 = 58.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.0090181, upper bound: 1.0090208

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5735
type: A, layer: 1, pos: 5735
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 4569
type: B, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 4560
type: A, layer: 1, pos: 4560
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982731, upper bound: 1.0081463
time: 5.07 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090092, upper bound: 1.0090130
time: 5.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.38 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 4, lower bound: -0.9982731, upper bound: 1.0081463
NS_B2, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 4, lower bound: -1.0090092, upper bound: 1.0090130

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -14.2564125, -11.0838490, -14.2392702, -11.0884342, -2.4459381, 2.4365983
1: -10.6055689, -7.9192228, -10.5863609, -7.9344049, -1.9506726, 1.9495695
2: -10.1383352, -7.3426933, -10.1246023, -7.3621340, -2.2606955, 2.2695022
3: -12.7662830, -10.3635712, -12.7497931, -10.3731203, -1.9064326, 1.8977575
4: 5.8982973, 8.4014158, 5.9232831, 8.3751211, -2.1198201, 2.1141963
5: -8.3564529, -5.7673421, -8.3391552, -5.7829990, -1.9162107, 1.9117794
6: -12.7058411, -9.7270794, -12.6938324, -9.7448006, -2.0902939, 2.0775120
7: -6.1839533, -3.3422699, -6.1540008, -3.3660150, -2.5171375, 2.5102911
8: -2.9951944, -0.2378821, -2.9860826, -0.2476339, -2.1831245, 2.1841516
9: -5.4291449, -3.2247987, -5.3939943, -3.2494354, -1.4686584, 1.4638314

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906209, upper bound: 1.0081365
time: 5.95 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982680, upper bound: 1.0081425
time: 5.80 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -14.2722244, -11.0764160, -14.2722216, -11.0764179, -2.4655313, 2.4934630
1: -10.6166239, -7.9022150, -10.6166210, -7.9022193, -2.0254941, 2.0394914
2: -10.1443129, -7.3213792, -10.1443110, -7.3213830, -2.3280783, 2.3304873
3: -12.7821159, -10.3563175, -12.7821140, -10.3563175, -1.9555011, 1.9442835
4: 5.8858547, 8.4309292, 5.8858566, 8.4309235, -2.2282734, 2.2495737
5: -8.3676167, -5.7517152, -8.3676138, -5.7517204, -1.9603539, 1.9654832
6: -12.7108383, -9.7072086, -12.7108364, -9.7072153, -2.2080932, 2.2138472
7: -6.2174854, -3.3342147, -6.2174788, -3.3342175, -2.7381716, 2.7233996
8: -3.0022955, -0.2282648, -3.0022936, -0.2282672, -2.2263432, 2.2275944
9: -5.4689369, -3.2161674, -5.4689260, -3.2161684, -1.6696615, 1.6465502

Time for backsubstitution: 22.74 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013772, upper bound: 1.0090067
time: 5.48 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090045, upper bound: 1.0090064
time: 4.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.04 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 33.04
Output dim: 4, lower bound: -0.9906209, upper bound: 1.0081365
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 33.04
Output dim: 4, lower bound: -0.9982680, upper bound: 1.0081425
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 33.04
Output dim: 4, lower bound: -1.0013772, upper bound: 1.0090067
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 33.04
Output dim: 4, lower bound: -1.0090045, upper bound: 1.0090064

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -14.2556725, -11.1332312, -14.2389765, -11.1081896, -2.4205360, 2.3823524
1: -10.6016436, -7.9228292, -10.5848007, -7.9358516, -1.9451499, 1.9441888
2: -10.1351862, -7.3626904, -10.1233530, -7.3701296, -2.2433734, 2.2432904
3: -12.7614098, -10.3651428, -12.7478390, -10.3737755, -1.9009900, 1.8943591
4: 5.9190731, 8.4001560, 5.9316063, 8.3746185, -2.0980172, 2.1044269
5: -8.3513508, -5.7718110, -8.3371048, -5.7847929, -1.9066868, 1.9031911
6: -12.7035723, -9.8006477, -12.6929283, -9.7742205, -2.0590582, 2.0032096
7: -6.1776590, -3.3436251, -6.1514668, -3.3665676, -2.5094995, 2.5052242
8: -2.9924488, -0.2517223, -2.9849815, -0.2531691, -2.1753273, 2.1693592
9: -5.4228287, -3.2268763, -5.3914638, -3.2502642, -1.4612966, 1.4589455

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9867600, upper bound: 1.0081265
time: 4.42 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906113, upper bound: 1.0081296
time: 4.63 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -14.4084148, -11.0696802, -14.2392683, -11.0884514, -2.4788470, 2.4371948
1: -10.6128473, -7.9093990, -10.5863562, -7.9344087, -1.9580414, 1.9591801
2: -10.1983423, -7.3366890, -10.1245995, -7.3621578, -2.2980776, 2.2727151
3: -12.7762375, -10.3543797, -12.7497864, -10.3731194, -1.9166527, 1.9065671
4: 5.8836532, 8.4555378, 5.9232998, 8.3751202, -2.1288347, 2.1273539
5: -8.3687315, -5.7626429, -8.3391514, -5.7830067, -1.9311881, 1.9145465
6: -12.9493828, -9.7223110, -12.6938305, -9.7448378, -2.1254611, 2.0499821
7: -6.2072954, -3.3363919, -6.1539927, -3.3660169, -2.5389643, 2.5153461
8: -3.0419846, -0.2325644, -2.9860792, -0.2476506, -2.2305355, 2.1856542
9: -5.4365687, -3.2009268, -5.3939853, -3.2494373, -1.4716735, 1.4857078

Time for backsubstitution: 22.60 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9944038, upper bound: 1.0081354
time: 4.98 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982585, upper bound: 1.0081316
time: 5.58 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -14.2714767, -11.1257534, -14.2719193, -11.0961571, -2.4401555, 2.4392753
1: -10.6126785, -7.9058185, -10.6150465, -7.9036622, -2.0199389, 2.0340667
2: -10.1411476, -7.3414040, -10.1430492, -7.3294039, -2.3107595, 2.3042674
3: -12.7772465, -10.3578901, -12.7801600, -10.3569546, -1.9500875, 1.9409022
4: 5.9066381, 8.4296646, 5.8941774, 8.4304218, -2.2064676, 2.2387404
5: -8.3625031, -5.7561922, -8.3655748, -5.7535191, -1.9508586, 1.9569173
6: -12.7085752, -9.7807713, -12.7099314, -9.7366304, -2.1768436, 2.1395702
7: -6.2112103, -3.3355656, -6.2149558, -3.3347645, -2.7305732, 2.7183437
8: -2.9995480, -0.2421093, -3.0011945, -0.2338052, -2.2184982, 2.2127872
9: -5.4626179, -3.2182465, -5.4663968, -3.2169962, -1.6623008, 1.6416702

Time for backsubstitution: 22.65 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9975048, upper bound: 1.0089961
time: 4.34 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013676, upper bound: 1.0089995
time: 4.21 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -14.4242496, -11.0621958, -14.2722216, -11.0764351, -2.5019150, 2.4941139
1: -10.6239052, -7.8924026, -10.6166182, -7.9022245, -2.0328889, 2.0490496
2: -10.2043295, -7.3153014, -10.1443090, -7.3214059, -2.3623338, 2.3335884
3: -12.7920523, -10.3471203, -12.7821054, -10.3563194, -1.9653254, 1.9530656
4: 5.8710942, 8.4850330, 5.8858747, 8.4309216, -2.2383823, 2.2550578
5: -8.3798876, -5.7470341, -8.3676090, -5.7517266, -1.9752879, 1.9682271
6: -12.9543886, -9.7024498, -12.7108345, -9.7072496, -2.2386246, 2.1863990
7: -6.2408156, -3.3283505, -6.2174711, -3.3342190, -2.7601109, 2.7284484
8: -3.0490837, -0.2229133, -3.0022922, -0.2282801, -2.2737708, 2.2290678
9: -5.4763184, -3.1923099, -5.4689164, -3.2161703, -1.6725698, 1.6633050

Time for backsubstitution: 22.68 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4569

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0051268, upper bound: 1.0089962
time: 3.93 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089949, upper bound: 1.0089995
time: 4.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.38 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -0.9867600, upper bound: 1.0081265
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -0.9906113, upper bound: 1.0081296
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -0.9944038, upper bound: 1.0081354
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -0.9982585, upper bound: 1.0081316
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -0.9975048, upper bound: 1.0089961
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -1.0013676, upper bound: 1.0089995
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -1.0051268, upper bound: 1.0089962
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.38
Output dim: 4, lower bound: -1.0089949, upper bound: 1.0089995

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14.2533007, -11.1352463, -14.2290916, -11.1165428, -2.4065008, 2.3665161
1: -10.5997820, -7.9247599, -10.5797405, -7.9456463, -1.9327989, 1.9372218
2: -10.1157885, -7.3631916, -10.0855160, -7.3916221, -2.1991401, 2.2049723
3: -12.7525539, -10.3664379, -12.7287350, -10.3876324, -1.8784480, 1.8738899
4: 5.9228382, 8.3880024, 5.9578705, 8.3533878, -2.0733223, 2.0658584
5: -8.3504009, -5.7739043, -8.3317547, -5.7968259, -1.8929033, 1.8921900
6: -12.7021236, -9.8026009, -12.6820707, -9.7797031, -2.0522547, 1.9901018
7: -6.1749825, -3.3557706, -6.1316366, -3.3898981, -2.4833713, 2.4734187
8: -2.9748116, -0.2527485, -2.9519515, -0.2770286, -2.1335497, 2.1355114
9: -5.4103575, -3.2275863, -5.3691740, -3.2658882, -1.4294548, 1.4364340

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of NS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_B1_A1_B1_B1

### Relational analysis result of NS_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9867604, upper bound: 1.0005038
time: 6.29 seconds

## Relational analysis of NS_B1_A1_B1_B2

### Relational analysis result of NS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9867600, upper bound: 1.0081265
time: 4.31 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14.2556667, -11.1332359, -14.2389727, -11.1081934, -2.4164925, 2.3835869
1: -10.6016407, -7.9228320, -10.5847931, -7.9358559, -1.9456725, 1.9440339
2: -10.1351671, -7.3626909, -10.1233177, -7.3701320, -2.2433491, 2.2101412
3: -12.7614002, -10.3651438, -12.7478218, -10.3737774, -1.9009819, 1.8818588
4: 5.9190779, 8.4001446, 5.9316134, 8.3745975, -2.0771041, 2.0976892
5: -8.3513489, -5.7718134, -8.3371029, -5.7847962, -1.9053183, 1.9015350
6: -12.7035742, -9.8006535, -12.6929255, -9.7742271, -2.0572515, 2.0032034
7: -6.1776557, -3.3436356, -6.1514606, -3.3665867, -2.4871793, 2.5052066
8: -2.9924259, -0.2517233, -2.9849429, -0.2531700, -2.1753073, 2.1406288
9: -5.4228191, -3.2268772, -5.3914490, -3.2502656, -1.4492941, 1.4342060

Time for backsubstitution: 22.76 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of NS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906113, upper bound: 1.0005052
time: 4.37 seconds

## Relational analysis of NS_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906113, upper bound: 1.0081296
time: 4.51 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -14.4060326, -11.0716639, -14.2293835, -11.0968208, -2.4648943, 2.4213395
1: -10.6109886, -7.9113426, -10.5812979, -7.9442029, -1.9456921, 1.9522114
2: -10.1789684, -7.3371944, -10.0867720, -7.3836589, -2.2516255, 2.2343895
3: -12.7673979, -10.3556938, -12.7306852, -10.3869991, -1.8941207, 1.8860905
4: 5.8873291, 8.4433832, 5.9496074, 8.3538876, -2.1040039, 2.0847068
5: -8.3678036, -5.7647367, -8.3338242, -5.7950363, -1.9173832, 1.9035373
6: -12.9479218, -9.7242556, -12.6829720, -9.7503119, -2.1186724, 2.0368898
7: -6.2046928, -3.3485217, -6.1341572, -3.3893461, -2.5127888, 2.4835539
8: -3.0243797, -0.2335930, -2.9530468, -0.2715139, -2.1853118, 2.1518173
9: -5.4241037, -3.2016501, -5.3716974, -3.2650676, -1.4398057, 1.4632189

Time for backsubstitution: 22.84 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9943922, upper bound: 1.0005027
time: 4.62 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9943933, upper bound: 1.0005037
time: 4.24 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -14.4084091, -11.0696812, -14.2392645, -11.0884552, -2.4710698, 2.4384542
1: -10.6128407, -7.9094005, -10.5863495, -7.9344139, -1.9585638, 1.9590259
2: -10.1983242, -7.3366909, -10.1245651, -7.3621569, -2.2792583, 2.2395904
3: -12.7762280, -10.3543797, -12.7497692, -10.3731241, -1.9166446, 1.8940659
4: 5.8836584, 8.4555254, 5.9233074, 8.3750963, -2.1079206, 2.1140425
5: -8.3687315, -5.7626457, -8.3391495, -5.7830100, -1.9298034, 1.9128909
6: -12.9493790, -9.7223129, -12.6938305, -9.7448406, -2.1210613, 2.0499763
7: -6.2072916, -3.3364012, -6.1539860, -3.3660345, -2.5166464, 2.5153303
8: -3.0419621, -0.2325649, -2.9860392, -0.2476521, -2.2130008, 2.1569238
9: -5.4365592, -3.2009277, -5.3939691, -3.2494383, -1.4596660, 1.4597809

Time for backsubstitution: 22.86 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 821

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982583, upper bound: 0.9997041
time: 6.26 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982585, upper bound: 1.0081316
time: 4.91 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -14.2690697, -11.1277819, -14.2619028, -11.1044960, -2.4260912, 2.4232130
1: -10.6108236, -7.9077482, -10.6100664, -7.9134588, -2.0075879, 2.0271840
2: -10.1217537, -7.3419209, -10.1052322, -7.3510070, -2.2664199, 2.2659159
3: -12.7683830, -10.3592186, -12.7610903, -10.3708916, -1.9273548, 1.9206519
4: 5.9104557, 8.4175072, 5.9206271, 8.4091930, -2.1815076, 2.1956143
5: -8.3615942, -5.7582846, -8.3602676, -5.7655239, -1.9370813, 1.9458477
6: -12.7071323, -9.7827063, -12.6990538, -9.7420712, -2.1701145, 2.1263771
7: -6.2085361, -3.3477018, -6.1951118, -3.3580470, -2.7044191, 2.6865005
8: -2.9819198, -0.2431402, -2.9681897, -0.2577047, -2.1767287, 2.1789250
9: -5.4501557, -3.2189598, -5.4441242, -3.2326460, -1.6304636, 1.6191812

Time for backsubstitution: 22.79 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9859323, upper bound: 1.0085496
time: 4.08 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974990, upper bound: 1.0089913
time: 3.79 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.2714748, -11.1257563, -14.2719135, -11.0961590, -2.4361172, 2.4405112
1: -10.6126747, -7.9058204, -10.6150427, -7.9036674, -2.0204611, 2.0339127
2: -10.1411285, -7.3414063, -10.1430140, -7.3294039, -2.3107347, 2.2711165
3: -12.7772350, -10.3578911, -12.7801437, -10.3569574, -1.9500790, 1.9284024
4: 5.9066429, 8.4296513, 5.8941841, 8.4303980, -2.1855540, 2.2253666
5: -8.3625011, -5.7561917, -8.3655739, -5.7535224, -1.9494901, 1.9552622
6: -12.7085724, -9.7807703, -12.7099295, -9.7366333, -2.1751318, 2.1395652
7: -6.2112060, -3.3355761, -6.2149496, -3.3347831, -2.7082486, 2.7183270
8: -2.9995270, -0.2421112, -3.0011549, -0.2338095, -2.2184782, 2.1840568
9: -5.4626102, -3.2182455, -5.4663806, -3.2169986, -1.6502781, 1.6169274

Time for backsubstitution: 22.88 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9897814, upper bound: 1.0085506
time: 6.32 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013619, upper bound: 1.0089914
time: 3.63 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.4218311, -11.0641947, -14.2622013, -11.0847902, -2.4878898, 2.4780393
1: -10.6220531, -7.8943472, -10.6116409, -7.9120188, -2.0205431, 2.0421243
2: -10.1849585, -7.3158193, -10.1064997, -7.3430200, -2.3158221, 2.2952354
3: -12.7832108, -10.3484583, -12.7630386, -10.3702660, -1.9426031, 1.9328051
4: 5.8748226, 8.4728737, 5.9123621, 8.4096966, -2.2133112, 2.2119064
5: -8.3789825, -5.7491274, -8.3623238, -5.7637310, -1.9614906, 1.9571486
6: -12.9529371, -9.7043810, -12.6999626, -9.7126846, -2.2318692, 2.1732230
7: -6.2382145, -3.3404729, -6.1976194, -3.3575006, -2.7339296, 2.6966205
8: -3.0314875, -0.2239470, -2.9692888, -0.2521849, -2.2267914, 2.1952171
9: -5.4638605, -3.1930351, -5.4466505, -3.2318258, -1.6407061, 1.6408346

Time for backsubstitution: 22.82 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9935678, upper bound: 1.0085594
time: 5.22 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0051211, upper bound: 1.0089916
time: 3.82 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.4242458, -11.0621967, -14.2722158, -11.0764370, -2.4941068, 2.4953761
1: -10.6239014, -7.8924060, -10.6166115, -7.9022274, -2.0334120, 2.0488958
2: -10.2043114, -7.3153014, -10.1442738, -7.3214068, -2.3434639, 2.3004627
3: -12.7920437, -10.3471184, -12.7820873, -10.3563213, -1.9653163, 1.9405642
4: 5.8710995, 8.4850216, 5.8858833, 8.4308987, -2.2174654, 2.2416592
5: -8.3798866, -5.7470360, -8.3676081, -5.7517309, -1.9706068, 1.9665720
6: -12.9543877, -9.7024536, -12.7108316, -9.7072515, -2.2342906, 2.1863937
7: -6.2408118, -3.3283606, -6.2174659, -3.3342378, -2.7377892, 2.7284341
8: -3.0490630, -0.2229137, -3.0022531, -0.2282858, -2.2544241, 2.2003384
9: -5.4763088, -3.1923113, -5.4689016, -3.2161722, -1.6605430, 1.6373703

Time for backsubstitution: 22.88 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974223, upper bound: 1.0085554
time: 4.65 seconds

## Relational analysis of NS_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089893, upper bound: 1.0089945
time: 4.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.01 seconds
NS_B1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9867604, upper bound: 1.0005038
NS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9867600, upper bound: 1.0081265
NS_B1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9906113, upper bound: 1.0005052
NS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9906113, upper bound: 1.0081296
NS_B1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9943922, upper bound: 1.0005027
NS_B1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9943933, upper bound: 1.0005037
NS_B1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9982583, upper bound: 0.9997041
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9982585, upper bound: 1.0081316
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9859323, upper bound: 1.0085496
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9974990, upper bound: 1.0089913
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9897814, upper bound: 1.0085506
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -1.0013619, upper bound: 1.0089914
NS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9935678, upper bound: 1.0085594
NS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -1.0051211, upper bound: 1.0089916
NS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -0.9974223, upper bound: 1.0085554
NS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.01
Output dim: 4, lower bound: -1.0089893, upper bound: 1.0089945

## BFS NS instance: NS_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -14.2533007, -11.1352463, -14.3813419, -11.0826035, -2.4373460, 2.4006362
1: -10.5997820, -7.9247599, -10.5886192, -7.9343858, -1.9439278, 1.9457402
2: -10.1157885, -7.3631916, -10.1468039, -7.3777256, -2.2116184, 2.2360752
3: -12.7525539, -10.3664379, -12.7407522, -10.3777685, -1.8878775, 1.8868139
4: 5.9228382, 8.3880024, 5.9349651, 8.4080286, -2.0947409, 2.0840797
5: -8.3504009, -5.7739043, -8.3459892, -5.7902870, -1.8979988, 1.9057879
6: -12.7021236, -9.8026009, -12.9264698, -9.7455387, -2.0827169, 2.0341377
7: -6.1749825, -3.3557706, -6.1575813, -3.3834057, -2.4894028, 2.4982190
8: -2.9748116, -0.2527485, -2.9998653, -0.2661963, -2.1444035, 2.1838479
9: -5.4103575, -3.2275863, -5.3791351, -3.2411642, -1.4369469, 1.4444954

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 821

### Candidate
type: B, layer: 1, pos: 821

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_B1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9867601, upper bound: 0.9996929
time: 5.14 seconds

## Relational analysis of NS_B1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9867601, upper bound: 1.0081267
time: 4.16 seconds

## BFS NS instance: NS_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -14.2556667, -11.1332359, -14.3912392, -11.0742769, -2.4472837, 2.4112399
1: -10.6016407, -7.9228320, -10.5936651, -7.9245825, -1.9568133, 1.9525485
2: -10.1351671, -7.3626909, -10.1845942, -7.3561935, -2.2558355, 2.2375176
3: -12.7614002, -10.3651438, -12.7597618, -10.3639183, -1.9104037, 1.8946998
4: 5.9190779, 8.4001446, 5.9087858, 8.4292326, -2.0955877, 2.1136813
5: -8.3513489, -5.7718134, -8.3513041, -5.7782660, -1.9104176, 1.9145062
6: -12.7035742, -9.8006535, -12.9373598, -9.7400484, -2.0850992, 2.0429919
7: -6.1776557, -3.3436356, -6.1772928, -3.3601379, -2.4931879, 2.5301118
8: -2.9924259, -0.2517233, -3.0328085, -0.2423530, -2.1861439, 2.1844347
9: -5.4228191, -3.2268772, -5.4014368, -3.2255406, -1.4567950, 1.4422045

Time for backsubstitution: 23.52 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.40 + 543.20 = 601.60 seconds
