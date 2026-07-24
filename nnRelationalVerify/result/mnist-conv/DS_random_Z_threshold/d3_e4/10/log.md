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
execution time: IAR + RelationalAnalysis = 25.61 + 34.02 = 59.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.0090181, upper bound: 1.0090208

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0069371, upper bound: 1.0081843
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081840, upper bound: 1.0069359
time: 4.07 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.04 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.04
Output dim: 4, lower bound: -1.0069371, upper bound: 1.0081843
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.04
Output dim: 4, lower bound: -1.0081840, upper bound: 1.0069359

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4699512, 2.4699602
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0266256, 2.0266306
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3283758, 2.3283629
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9452653, 1.9452586
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2497330, 2.2497368
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9608331, 1.9608395
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2138100, 2.2138152
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7246599, 2.7246571
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2276306, 2.2276263
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6726465, 1.6726456

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0069321, upper bound: 1.0011394
time: 9.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9999015, upper bound: 1.0081779
time: 4.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4699607, 2.4699507
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0266304, 2.0266256
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3283634, 2.3283756
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9452586, 1.9452651
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2497368, 2.2497334
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9608397, 1.9608328
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2138152, 2.2138100
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7246580, 2.7246604
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2276268, 2.2276311
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6726456, 1.6726465

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0005509, upper bound: 1.0069318
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081797, upper bound: 0.9993132
time: 5.62 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.90 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 31.90
Output dim: 4, lower bound: -1.0069321, upper bound: 1.0011394
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.90
Output dim: 4, lower bound: -0.9999015, upper bound: 1.0081779
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 31.90
Output dim: 4, lower bound: -1.0005509, upper bound: 1.0069318
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.90
Output dim: 4, lower bound: -1.0081797, upper bound: 0.9993132

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4695015, 2.4692111
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0290852, 2.0286477
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3301625, 2.3298264
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9463582, 1.9465919
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2435136, 2.2442932
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9614468, 1.9613423
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2127848, 2.2129188
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7250204, 2.7250962
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2274418, 2.2274604
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6663394, 1.6654375

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9991080, upper bound: 1.0079447
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9996630, upper bound: 1.0073869
time: 8.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4217691, 2.4475346
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0265646, 2.0265021
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3088665, 2.3193047
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9446311, 1.9439178
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2446890, 2.2388892
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9603000, 1.9596751
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1442647, 2.1814373
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7242393, 2.7237649
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2194643, 2.2238283
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6702662, 1.6675415

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 821

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081748, upper bound: 0.9922719
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0011355, upper bound: 0.9993085
time: 4.24 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.27 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 4, lower bound: -0.9991080, upper bound: 1.0079447
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 4, lower bound: -0.9996630, upper bound: 1.0073869
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.27
Output dim: 4, lower bound: -1.0081748, upper bound: 0.9922719
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.27
Output dim: 4, lower bound: -1.0011355, upper bound: 0.9993085

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4771461, 2.4757957
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0287151, 2.0278180
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3301535, 2.3298228
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9481635, 1.9486899
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2438464, 2.2445803
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9602575, 1.9599805
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2138467, 2.2138371
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7205739, 2.7212157
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2294712, 2.2292075
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6675630, 1.6664932

Time for backsubstitution: 21.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9914758, upper bound: 1.0079404
time: 6.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9991035, upper bound: 1.0003030
time: 7.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4760876, 2.4768548
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0282555, 2.0282779
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3301592, 2.3298171
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9484558, 1.9483976
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2438006, 2.2446260
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9600849, 1.9601529
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2137036, 2.2139802
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7211404, 2.7206502
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2291889, 2.2294908
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6673951, 1.6666608

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4560

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9995328, upper bound: 1.0073865
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9996628, upper bound: 1.0072595
time: 6.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4210196, 2.4470849
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0285821, 2.0289617
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3103313, 2.3210917
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9459648, 1.9450109
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2392459, 2.2326703
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9608026, 1.9602890
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1433678, 2.1804118
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7246790, 2.7241244
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2192993, 2.2236395
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6630564, 1.6612339

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5735

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081670, upper bound: 0.9883858
time: 11.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9972553, upper bound: 0.9884076
time: 4.87 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 38.76 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 38.76
Output dim: 4, lower bound: -0.9914758, upper bound: 1.0079404
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 38.76
Output dim: 4, lower bound: -0.9991035, upper bound: 1.0003030
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 38.76
Output dim: 4, lower bound: -0.9995328, upper bound: 1.0073865
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 38.76
Output dim: 4, lower bound: -0.9996628, upper bound: 1.0072595
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 38.76
Output dim: 4, lower bound: -1.0081670, upper bound: 0.9883858
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 38.76
Output dim: 4, lower bound: -0.9972553, upper bound: 0.9884076

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4547291, 2.4276047
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0285912, 2.0277514
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3210821, 2.3103275
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9468164, 1.9480624
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2330022, 2.2395320
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9590993, 1.9594402
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1814742, 2.1442866
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7196774, 2.7207966
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2256689, 2.2210455
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6624575, 1.6641128

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 494

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9910582, upper bound: 1.0079391
time: 5.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9910887, upper bound: 1.0069660
time: 6.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4610653, 2.4658036
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0225997, 2.0205903
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3257823, 2.3238671
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9467492, 1.9460664
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2406926, 2.2404122
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9543929, 1.9524078
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2127614, 2.2132993
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7179570, 2.7162976
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2199116, 2.2168751
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6569357, 1.6589644

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9918998, upper bound: 1.0073846
time: 4.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9995284, upper bound: 0.9997559
time: 4.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4650364, 2.4618330
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0205674, 2.0226231
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3242087, 2.3254414
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9461246, 1.9466906
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2395864, 2.2415190
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9523401, 1.9544609
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.2130227, 2.2130380
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7167878, 2.7174687
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2165737, 2.2202144
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6596990, 1.6562014

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9920298, upper bound: 1.0072549
time: 10.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9996584, upper bound: 0.9996274
time: 7.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4430909, 2.4726987
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0396404, 2.0417950
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3122320, 2.3232980
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9562492, 1.9538751
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2178173, 2.2081838
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9648557, 1.9649911
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1376157, 2.1738386
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7382193, 2.7357936
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2180486, 2.2222095
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6332521, 1.6351545

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0073745, upper bound: 0.9881460
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0079302, upper bound: 0.9875938
time: 5.47 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.39 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.39
Output dim: 4, lower bound: -0.9910582, upper bound: 1.0079391
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.39
Output dim: 4, lower bound: -0.9910887, upper bound: 1.0069660
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.39
Output dim: 4, lower bound: -0.9918998, upper bound: 1.0073846
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.39
Output dim: 4, lower bound: -0.9995284, upper bound: 0.9997559
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.39
Output dim: 4, lower bound: -0.9920298, upper bound: 1.0072549
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 32.39
Output dim: 4, lower bound: -0.9996584, upper bound: 0.9996274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.39
Output dim: 4, lower bound: -1.0073745, upper bound: 0.9881460
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.39
Output dim: 4, lower bound: -1.0079302, upper bound: 0.9875938

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4323440, 2.4080257
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0319562, 2.0324538
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3287134, 2.3169556
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9472508, 1.9485548
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2274923, 2.2347107
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9426446, 1.9406347
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1515617, 2.1181121
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7154713, 2.7171173
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2175255, 2.2139168
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6491675, 1.6535752

Time for backsubstitution: 20.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4560

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9909275, upper bound: 1.0079362
time: 7.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9910580, upper bound: 1.0078089
time: 4.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4386501, 2.4176126
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0224757, 2.0205235
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3167133, 2.3043721
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9454021, 1.9454389
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2298484, 2.2353640
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9532351, 1.9518676
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1803889, 2.1437488
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7170615, 2.7158794
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2161102, 2.2087140
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6518307, 1.6565845

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9918951, upper bound: 1.0031578
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9876818, upper bound: 1.0073777
time: 4.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4426193, 2.4136419
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0204434, 2.0225563
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3151388, 2.3059464
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9447780, 1.9460635
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2287431, 2.2364707
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9511819, 1.9539208
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1806502, 2.1434875
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7158904, 2.7170501
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2127714, 2.2120533
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6545939, 1.6538215

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5735
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5735

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9881620, upper bound: 0.9963378
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9881431, upper bound: 1.0072477
time: 4.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4507360, 2.4792848
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0392709, 2.0409658
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3122225, 2.3232934
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9580536, 1.9559720
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2181497, 2.2084708
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9636655, 1.9636290
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1386771, 2.1747565
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7337756, 2.7319155
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2200775, 2.2239552
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6344781, 1.6362126

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0072020, upper bound: 0.9851981
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0044232, upper bound: 0.9879768
time: 4.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.2722282, -11.0764141, -14.2722282, -11.0764141, -2.4496765, 2.4803438
1: -10.6166239, -7.9022141, -10.6166239, -7.9022141, -2.0388112, 2.0414257
2: -10.1443138, -7.3213749, -10.1443138, -7.3213749, -2.3122282, 2.3232877
3: -12.7821198, -10.3563147, -12.7821198, -10.3563147, -1.9583464, 1.9556794
4: 5.8858533, 8.4309311, 5.8858533, 8.4309311, -2.2181039, 2.2085166
5: -8.3676195, -5.7517128, -8.3676195, -5.7517128, -1.9634933, 1.9638014
6: -12.7108393, -9.7072086, -12.7108393, -9.7072086, -2.1385336, 2.1748998
7: -6.2174892, -3.3342144, -6.2174892, -3.3342144, -2.7343402, 2.7313499
8: -3.0022974, -0.2282639, -3.0022974, -0.2282639, -2.2197943, 2.2242384
9: -5.4689426, -3.2161660, -5.4689426, -3.2161660, -1.6343102, 1.6363802

Time for backsubstitution: 21.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 494
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 4560
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 494

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0069558, upper bound: 0.9872053
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0079289, upper bound: 0.9871782
time: 6.75 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 32.48 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 32.48
Output dim: 4, lower bound: -0.9909275, upper bound: 1.0079362
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 32.48
Output dim: 4, lower bound: -0.9910580, upper bound: 1.0078089
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 32.48
Output dim: 4, lower bound: -0.9918951, upper bound: 1.0031578
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 32.48
Output dim: 4, lower bound: -0.9876818, upper bound: 1.0073777
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 32.48
Output dim: 4, lower bound: -0.9881620, upper bound: 0.9963378
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 32.48
Output dim: 4, lower bound: -0.9881431, upper bound: 1.0072477
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 32.48
Output dim: 4, lower bound: -1.0072020, upper bound: 0.9851981
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 32.48
Output dim: 4, lower bound: -1.0044232, upper bound: 0.9879768
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 32.48
Output dim: 4, lower bound: -1.0069558, upper bound: 0.9872053
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 32.48
Output dim: 4, lower bound: -1.0079289, upper bound: 0.9871782

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.63 + 542.51 = 602.14 seconds
