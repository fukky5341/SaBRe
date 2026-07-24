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
execution time: IAR + RelationalAnalysis = 25.00 + 35.32 = 60.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.0090181, upper bound: 1.0090208

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013861, upper bound: 1.0090140
time: 5.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090135, upper bound: 1.0090167
time: 4.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.47 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.47
Output dim: 4, lower bound: -1.0013861, upper bound: 1.0090140
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.47
Output dim: 4, lower bound: -1.0090135, upper bound: 1.0090167

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -14.2714777, -11.1257515, -14.2719250, -11.0961514, -2.4445662, 2.4157271
1: -10.6126785, -7.9058161, -10.6150532, -7.9036570, -2.0211110, 2.0212431
2: -10.1411486, -7.3414030, -10.1430521, -7.3293962, -2.3109674, 2.3020663
3: -12.7772474, -10.3578892, -12.7801666, -10.3569527, -1.9398088, 1.9418397
4: 5.9066372, 8.4296684, 5.8941746, 8.4304304, -2.2279062, 2.2398758
5: -8.3625050, -5.7561893, -8.3655796, -5.7535124, -1.9512920, 1.9522226
6: -12.7085743, -9.7807655, -12.7099323, -9.7366219, -2.1826043, 2.1395750
7: -6.2112145, -3.3355644, -6.2149658, -3.3347621, -2.7170372, 2.7195787
8: -2.9995489, -0.2421088, -3.0011954, -0.2338033, -2.2197528, 2.2127919
9: -5.4626245, -3.2182441, -5.4664140, -3.2169948, -1.6653032, 1.6677594

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906209, upper bound: 1.0081365
time: 6.26 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013772, upper bound: 1.0090067
time: 5.76 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -14.4242496, -11.0621929, -14.2722282, -11.0764313, -2.5048852, 2.4705610
1: -10.6239071, -7.8924022, -10.6166201, -7.9022179, -2.0340624, 2.0362220
2: -10.2043304, -7.3152962, -10.1443110, -7.3213978, -2.3655815, 2.3313885
3: -12.7920580, -10.3471165, -12.7821131, -10.3563166, -1.9550467, 1.9540031
4: 5.8710938, 8.4850359, 5.8858700, 8.4309311, -2.2598181, 2.2690897
5: -8.3798876, -5.7470312, -8.3676128, -5.7517228, -1.9757209, 1.9635315
6: -12.9543867, -9.7024460, -12.7108345, -9.7072420, -2.2460985, 2.1864047
7: -6.2408204, -3.3283491, -6.2174826, -3.3342173, -2.7465758, 2.7296844
8: -3.0490847, -0.2229114, -3.0022936, -0.2282796, -2.2750254, 2.2290726
9: -5.4763241, -3.1923094, -5.4689336, -3.2161684, -1.6756997, 1.6944265

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5735

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982680, upper bound: 1.0081425
time: 6.08 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090045, upper bound: 1.0090064
time: 4.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.47 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.47
Output dim: 4, lower bound: -0.9906209, upper bound: 1.0081365
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.47
Output dim: 4, lower bound: -1.0013772, upper bound: 1.0090067
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.47
Output dim: 4, lower bound: -0.9982680, upper bound: 1.0081425
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.47
Output dim: 4, lower bound: -1.0090045, upper bound: 1.0090064

## BFS NS instance: NS_A1_B1

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

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0042721
time: 4.36 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0081261
time: 4.42 seconds

## BFS NS instance: NS_A1_B2

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

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0051278
time: 4.31 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0089994
time: 4.57 seconds

## BFS NS instance: NS_A2_B1

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

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982584, upper bound: 1.0042808
time: 5.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982586, upper bound: 1.0081313
time: 4.94 seconds

## BFS NS instance: NS_A2_B2

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

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4569
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4569

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089952, upper bound: 1.0051287
time: 5.01 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089951, upper bound: 1.0089966
time: 4.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.22 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.22
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0042721
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.22
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0081261
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.22
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0051278
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.22
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0089994
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.22
Output dim: 4, lower bound: -0.9982584, upper bound: 1.0042808
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.22
Output dim: 4, lower bound: -0.9982586, upper bound: 1.0081313
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.22
Output dim: 4, lower bound: -1.0089952, upper bound: 1.0051287
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.22
Output dim: 4, lower bound: -1.0089951, upper bound: 1.0089966

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -14.2556686, -11.1332340, -14.2389736, -11.1081886, -2.4217544, 2.3782988
1: -10.6016378, -7.9228330, -10.5847950, -7.9358540, -1.9449954, 1.9447117
2: -10.1351519, -7.3626924, -10.1233330, -7.3701315, -2.2101922, 2.2432666
3: -12.7613945, -10.3651438, -12.7478275, -10.3737755, -1.8884902, 1.8943510
4: 5.9190817, 8.4001312, 5.9316106, 8.3746071, -2.0980043, 2.0835133
5: -8.3513470, -5.7718148, -8.3371038, -5.7847958, -1.9050331, 1.9018214
6: -12.7035742, -9.8006516, -12.6929274, -9.7742233, -2.0590529, 2.0014324
7: -6.1776528, -3.3436441, -6.1514626, -3.3665769, -2.5039997, 2.4829063
8: -2.9924088, -0.2517242, -2.9849596, -0.2531691, -2.1465983, 2.1693382
9: -5.4228129, -3.2268782, -5.3914552, -3.2502656, -1.4353783, 1.4589360

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0005034
time: 4.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0081263
time: 4.35 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.2714729, -11.1257572, -14.2719164, -11.0961561, -2.4413614, 2.4352179
1: -10.6126719, -7.9058228, -10.6150436, -7.9036655, -2.0197840, 2.0345898
2: -10.1411123, -7.3414059, -10.1430311, -7.3294039, -2.2775717, 2.3042431
3: -12.7772284, -10.3578920, -12.7801495, -10.3569555, -1.9375863, 1.9408941
4: 5.9066448, 8.4296417, 5.8941813, 8.4304085, -2.2032795, 2.2148662
5: -8.3625011, -5.7561932, -8.3655739, -5.7535205, -1.9492030, 1.9551291
6: -12.7085733, -9.7807713, -12.7099285, -9.7366314, -2.1768389, 2.1378458
7: -6.2112050, -3.3355844, -6.2149534, -3.3347731, -2.7231817, 2.6960201
8: -2.9995098, -0.2421122, -3.0011725, -0.2338085, -2.1897674, 2.2127686
9: -5.4626036, -3.2182465, -5.4663887, -3.2169976, -1.6363709, 1.6394455

Time for backsubstitution: 22.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0013687
time: 5.96 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0089994
time: 4.70 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -14.4084091, -11.0696840, -14.2392693, -11.0884523, -2.4736004, 2.4331264
1: -10.6128435, -7.9094028, -10.5863523, -7.9344120, -1.9578867, 1.9597018
2: -10.1983089, -7.3366895, -10.1245823, -7.3621573, -2.2611413, 2.2681291
3: -12.7762213, -10.3543816, -12.7497768, -10.3731232, -1.9041529, 1.9065585
4: 5.8836613, 8.4555130, 5.9233036, 8.3751059, -2.1288218, 2.1035204
5: -8.3687296, -5.7626472, -8.3391495, -5.7830091, -1.9295330, 1.9131780
6: -12.9493771, -9.7223167, -12.6938295, -9.7448387, -2.1207809, 2.0481858
7: -6.2072892, -3.3364110, -6.1539888, -3.3660264, -2.5332990, 2.4930234
8: -3.0419459, -0.2325664, -2.9860568, -0.2476521, -2.1988602, 2.1856337
9: -5.4365525, -3.2009277, -5.3939772, -3.2494397, -1.4456835, 1.4736810

Time for backsubstitution: 22.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9940302, upper bound: 1.0081265
time: 4.60 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982538, upper bound: 1.0081267
time: 5.20 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.4142189, -11.0705118, -14.2698050, -11.0784731, -2.4859171, 2.4801145
1: -10.6189327, -7.9022174, -10.6147661, -7.9041510, -2.0260124, 2.0366545
2: -10.1665249, -7.3369656, -10.1249208, -7.3219266, -2.3239021, 2.2892716
3: -12.7730570, -10.3610401, -12.7732410, -10.3576574, -1.9451375, 1.9303401
4: 5.8974581, 8.4638119, 5.8897042, 8.4187622, -2.1991148, 2.2301345
5: -8.3746328, -5.7590327, -8.3667145, -5.7538214, -1.9642639, 1.9544318
6: -12.9434757, -9.7079029, -12.7093983, -9.7091789, -2.2251186, 2.1796601
7: -6.2210875, -3.3515921, -6.2148027, -3.3463542, -2.7282195, 2.7023506
8: -3.0161302, -0.2468286, -2.9846644, -0.2293129, -2.2399368, 2.1873207
9: -5.4540224, -3.2079611, -5.4564638, -3.2168903, -1.6500430, 1.6314549

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9974224, upper bound: 1.0047052
time: 5.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089894, upper bound: 1.0051263
time: 4.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.4242420, -11.0621986, -14.2722187, -11.0764380, -2.4966116, 2.4900455
1: -10.6238995, -7.8924079, -10.6166134, -7.9022245, -2.0327353, 2.0495720
2: -10.2042961, -7.3153000, -10.1442881, -7.3214054, -2.3253589, 2.3310516
3: -12.7920361, -10.3471203, -12.7820950, -10.3563213, -1.9528246, 1.9530568
4: 5.8711023, 8.4850092, 5.8858786, 8.4309082, -2.2349906, 2.2311659
5: -8.3798847, -5.7470388, -8.3676090, -5.7517300, -1.9712570, 1.9664974
6: -12.9543829, -9.7024527, -12.7108326, -9.7072544, -2.2339544, 2.1846561
7: -6.2408113, -3.3283687, -6.2174668, -3.3342302, -2.7525840, 2.7061253
8: -3.0490460, -0.2229142, -3.0022697, -0.2282853, -2.2402949, 2.2290492
9: -5.4763041, -3.1923108, -5.4689083, -3.2161713, -1.6465688, 1.6512694

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 4569
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 494
type: B, layer: 1, pos: 4560
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 821

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9974223, upper bound: 1.0085552
time: 4.59 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089894, upper bound: 1.0089944
time: 4.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.96 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.96
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0005034
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0081263
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.96
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0013687
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -1.0013677, upper bound: 1.0089994
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -0.9940302, upper bound: 1.0081265
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -0.9982538, upper bound: 1.0081267
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.96
Output dim: 4, lower bound: -0.9974224, upper bound: 1.0047052
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -1.0089894, upper bound: 1.0051263
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -0.9974223, upper bound: 1.0085552
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 4, lower bound: -1.0089894, upper bound: 1.0089944

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -14.2556686, -11.1332340, -14.3912401, -11.0742741, -2.4526320, 2.4086621
1: -10.6016378, -7.9228330, -10.5936680, -7.9245820, -1.9561367, 1.9532244
2: -10.1351519, -7.3626924, -10.1846075, -7.3561921, -2.2226787, 2.2556353
3: -12.7613945, -10.3651438, -12.7597704, -10.3639174, -1.8979120, 1.9071915
4: 5.9190817, 8.4001312, 5.9087830, 8.4292450, -2.1061192, 2.1031163
5: -8.3513470, -5.7718148, -8.3513050, -5.7782650, -1.9101305, 1.9138179
6: -12.7035742, -9.8006516, -12.9373608, -9.7400475, -2.0848546, 2.0432823
7: -6.1776528, -3.3436441, -6.1772943, -3.3601294, -2.5099425, 2.5078115
8: -2.9924088, -0.2517242, -3.0328257, -0.2423501, -2.1574345, 2.1985731
9: -5.4228129, -3.2268782, -5.4014430, -3.2255392, -1.4428844, 1.4670489

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 821

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906112, upper bound: 0.9996925
time: 4.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9906114, upper bound: 1.0081295
time: 4.41 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.2714729, -11.1257572, -14.4242392, -11.0621996, -2.4722457, 2.4633222
1: -10.6126719, -7.9058228, -10.6238976, -7.8924103, -2.0308723, 2.0431395
2: -10.1411123, -7.3414059, -10.2043085, -7.3153162, -2.2900829, 2.3184912
3: -12.7772284, -10.3578920, -12.7920380, -10.3471375, -1.9469457, 1.9532804
4: 5.9066448, 8.4296417, 5.8711028, 8.4850159, -2.2092676, 2.2320786
5: -8.3625011, -5.7561932, -8.3798809, -5.7470403, -1.9542365, 1.9648039
6: -12.7085733, -9.7807713, -12.9543829, -9.7024574, -2.1979575, 2.1627233
7: -6.2112050, -3.3355844, -6.2408023, -3.3283694, -2.7291465, 2.7210450
8: -2.9995098, -0.2421122, -3.0490503, -0.2229156, -2.2005873, 2.2404590
9: -5.4626036, -3.2182465, -5.4762983, -3.1923137, -1.6438713, 1.6451039

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906112, upper bound: 0.9982467
time: 4.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9906114, upper bound: 0.9982446
time: 4.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -14.4005060, -11.0772095, -14.2238626, -11.1032343, -2.4682951, 2.4263554
1: -10.5841408, -7.9126263, -10.5325184, -7.9533238, -1.8840494, 1.8834534
2: -10.1896057, -7.3569140, -10.1005678, -7.4019990, -2.2039299, 2.2015653
3: -12.7626543, -10.3594780, -12.7203064, -10.3855801, -1.7903490, 1.7881126
4: 5.8922186, 8.4377680, 5.9471827, 8.3412523, -1.7589085, 1.7246952
5: -8.3552980, -5.7683234, -8.3116169, -5.7977719, -1.7751191, 1.7527118
6: -12.9416094, -9.7438946, -12.6729012, -9.7835426, -1.9211583, 1.8630273
7: -6.1973948, -3.3615308, -6.1260753, -3.4143682, -2.3708010, 2.3326592
8: -3.0098219, -0.2408037, -2.9223409, -0.2752366, -2.1297703, 2.1101718
9: -5.4316654, -3.2067347, -5.3843522, -3.2612829, -1.2289310, 1.2565980

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 821

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9940300, upper bound: 0.9996993
time: 4.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9940302, upper bound: 1.0081264
time: 4.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -14.4084082, -11.0696831, -14.2392654, -11.0884590, -2.4728985, 2.4406900
1: -10.6128368, -7.9094028, -10.5863447, -7.9344130, -1.9577699, 1.9189255
2: -10.1983099, -7.3366919, -10.1245785, -7.3621616, -2.2390723, 2.2615857
3: -12.7762184, -10.3543835, -12.7497711, -10.3731251, -1.9037619, 1.8935413
4: 5.8836622, 8.4555120, 5.9233074, 8.3751001, -2.1245809, 2.0965776
5: -8.3687267, -5.7626495, -8.3391438, -5.7830114, -1.9275298, 1.9130511
6: -12.9493799, -9.7223206, -12.6938248, -9.7448521, -2.1103973, 2.0481818
7: -6.2072887, -3.3364139, -6.1539860, -3.3660359, -2.4994688, 2.4924898
8: -3.0419416, -0.2325668, -2.9860468, -0.2476530, -2.1863980, 2.1549854
9: -5.4365530, -3.2009277, -5.3939748, -3.2494411, -1.4441013, 1.4804897

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 821

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9982536, upper bound: 0.9996998
time: 6.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982538, upper bound: 1.0081267
time: 7.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -14.4136839, -11.0705147, -14.2688770, -11.0784750, -2.4845071, 2.4817505
1: -10.6189308, -7.9022341, -10.6147633, -7.9041839, -2.0250759, 2.0390933
2: -10.1665249, -7.3369813, -10.1249180, -7.3219543, -2.3208346, 2.2910399
3: -12.7729998, -10.3610420, -12.7731419, -10.3576612, -1.9464211, 1.9297626
4: 5.8974638, 8.4638081, 5.8897133, 8.4187546, -2.1936607, 2.2202830
5: -8.3746290, -5.7590628, -8.3667097, -5.7538757, -1.9637709, 1.9550197
6: -12.9434328, -9.7079096, -12.7093239, -9.7091894, -2.2194653, 2.1796110
7: -6.2210817, -3.3516154, -6.2147961, -3.3463972, -2.7203393, 2.7021661
8: -3.0161119, -0.2468300, -2.9846334, -0.2293196, -2.2375050, 2.1872940
9: -5.4540138, -3.2079630, -5.4564495, -3.2168941, -1.6369319, 1.6180451

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 494
type: A, layer: 1, pos: 4560
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 147

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089847, upper bound: 1.0009060
time: 3.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089847, upper bound: 1.0051215
time: 4.67 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 60.32 + 540.28 = 600.60 seconds
