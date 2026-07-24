## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.184724793


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.2142577, -4.4942431, -5.2142577, -4.4942431, -0.3232119, 0.3232119)
1: (-8.6666365, -7.9502511, -8.6666365, -7.9502511, -0.3898010, 0.3898010)
2: (-0.8758705, -0.2017303, -0.8758705, -0.2017303, -0.3374567, 0.3374565)
3: (-4.4505510, -3.5619974, -4.4505510, -3.5619974, -0.4445329, 0.4445329)
4: (-11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3013635, 0.3013635)
5: (-9.1612968, -8.2840576, -9.1612968, -8.2840576, -0.3966613, 0.3966613)
6: (-11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4207950, 0.4207947)
7: (-11.7566013, -10.9271116, -11.7566013, -10.9271116, -0.3741808, 0.3741808)
8: (9.1338654, 9.8127308, 9.1338654, 9.8127308, -0.4533181, 0.4533181)
9: (-5.4181666, -4.7633429, -5.4181666, -4.7633429, -0.2832100, 0.2832100)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.84 + 35.79 = 57.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1865907, upper bound: 0.1865909

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 499

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858655, upper bound: 0.1865824
time: 5.07 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865890, upper bound: 0.1865895
time: 3.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.02 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.02
Output dim: 8, lower bound: -0.1858655, upper bound: 0.1865824
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.02
Output dim: 8, lower bound: -0.1865890, upper bound: 0.1865895

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.2128687, -4.5120316, -5.2142563, -4.4988585, -0.3128529, 0.3054235
1: -8.6634064, -7.9507961, -8.6658020, -7.9503174, -0.3864508, 0.3883209
2: -0.8755257, -0.2071171, -0.8758700, -0.2031248, -0.3357329, 0.3321660
3: -4.4375553, -3.5630407, -4.4471774, -3.5619969, -0.4316273, 0.4400964
4: -11.9245024, -11.1505375, -11.9263439, -11.1403580, -0.2942177, 0.2873712
5: -9.1416817, -8.2856779, -9.1562014, -8.2840691, -0.3771143, 0.3835943
6: -11.1090651, -10.2404785, -11.1244717, -10.2386999, -0.3999343, 0.4125853
7: -11.7561760, -10.9320564, -11.7565985, -10.9283943, -0.3724189, 0.3692439
8: 9.1385126, 9.8123331, 9.1350603, 9.8127308, -0.4486313, 0.4517131
9: -5.4154100, -4.7828226, -5.4178696, -4.7683988, -0.2677773, 0.2633510

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 499

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858655, upper bound: 0.1858657
time: 4.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858655, upper bound: 0.1865824
time: 6.78 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.2142577, -4.4942460, -5.2142577, -4.4942431, -0.3232126, 0.3083048
1: -8.6666374, -7.9502521, -8.6666365, -7.9502511, -0.3872299, 0.3898010
2: -0.8758705, -0.2017304, -0.8758705, -0.2017303, -0.3374567, 0.3339980
3: -4.4505491, -3.5619974, -4.4505510, -3.5619974, -0.4332209, 0.4445329
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.3013453, 0.2893162
5: -9.1612968, -8.2840586, -9.1612968, -8.2840576, -0.3793964, 0.3966613
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4020977, 0.4207947
7: -11.7566013, -10.9271126, -11.7566013, -10.9271116, -0.3741775, 0.3698542
8: 9.1338673, 9.8127308, 9.1338654, 9.8127308, -0.4495106, 0.4533181
9: -5.4181666, -4.7633448, -5.4181666, -4.7633429, -0.2831869, 0.2660084

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 499

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865822, upper bound: 0.1858657
time: 4.63 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865821, upper bound: 0.1858657
time: 3.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.68 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.68
Output dim: 8, lower bound: -0.1858655, upper bound: 0.1858657
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.68
Output dim: 8, lower bound: -0.1858655, upper bound: 0.1865824
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.68
Output dim: 8, lower bound: -0.1865822, upper bound: 0.1858657
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.68
Output dim: 8, lower bound: -0.1865821, upper bound: 0.1858657

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.2128687, -4.5120316, -5.2128687, -4.5120316, -0.3040264, 0.3040264
1: -8.6634064, -7.9507961, -8.6634064, -7.9507961, -0.3859024, 0.3859024
2: -0.8755257, -0.2071171, -0.8755257, -0.2071171, -0.3318181, 0.3318176
3: -4.4375553, -3.5630407, -4.4375553, -3.5630407, -0.4305692, 0.4305692
4: -11.9245024, -11.1505375, -11.9245024, -11.1505375, -0.2852931, 0.2852933
5: -9.1416817, -8.2856779, -9.1416817, -8.2856779, -0.3754835, 0.3754835
6: -11.1090651, -10.2404785, -11.1090651, -10.2404785, -0.3981276, 0.3981278
7: -11.7561760, -10.9320564, -11.7561760, -10.9320564, -0.3688111, 0.3688109
8: 9.1385126, 9.8123331, 9.1385126, 9.8123331, -0.4482317, 0.4482317
9: -5.4154100, -4.7828226, -5.4154100, -4.7828226, -0.2604644, 0.2604644

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 63

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1858603
time: 5.30 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1858603
time: 5.58 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.2128687, -4.5120316, -5.2142577, -4.4942460, -0.3131623, 0.3054261
1: -8.6634064, -7.9507961, -8.6666374, -7.9502521, -0.3865371, 0.3891664
2: -0.8755257, -0.2071171, -0.8758705, -0.2017304, -0.3371077, 0.3321664
3: -4.4375553, -3.5630407, -4.4505491, -3.5619974, -0.4316278, 0.4404662
4: -11.9245024, -11.1505375, -11.9265947, -11.1367950, -0.2942255, 0.2876732
5: -9.1416817, -8.2856779, -9.1612968, -8.2840586, -0.3771234, 0.3837006
6: -11.1090651, -10.2404785, -11.1298752, -10.2386789, -0.3999581, 0.4126146
7: -11.7561760, -10.9320564, -11.7566013, -10.9271126, -0.3737426, 0.3692458
8: 9.1385126, 9.8123331, 9.1338673, 9.8127308, -0.4486313, 0.4529185
9: -5.4154100, -4.7828226, -5.4181666, -4.7633448, -0.2677788, 0.2637656

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 94

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 63

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1865767
time: 6.50 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1865764
time: 7.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.2142577, -4.4942460, -5.2128687, -4.5120316, -0.3054261, 0.3131623
1: -8.6666374, -7.9502521, -8.6634064, -7.9507961, -0.3891659, 0.3865371
2: -0.8758705, -0.2017304, -0.8755257, -0.2071171, -0.3321667, 0.3371079
3: -4.4505491, -3.5619974, -4.4375553, -3.5630407, -0.4404662, 0.4316278
4: -11.9265947, -11.1367950, -11.9245024, -11.1505375, -0.2876732, 0.2942255
5: -9.1612968, -8.2840586, -9.1416817, -8.2856779, -0.3837006, 0.3771234
6: -11.1298752, -10.2386789, -11.1090651, -10.2404785, -0.4126146, 0.3999584
7: -11.7566013, -10.9271126, -11.7561760, -10.9320564, -0.3692460, 0.3737426
8: 9.1338673, 9.8127308, 9.1385126, 9.8123331, -0.4529185, 0.4486313
9: -5.4181666, -4.7633448, -5.4154100, -4.7828226, -0.2637656, 0.2677788

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 63

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862132, upper bound: 0.1858600
time: 6.98 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865762, upper bound: 0.1858600
time: 5.21 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.2142577, -4.4942460, -5.2142577, -4.4942460, -0.3083048, 0.3083048
1: -8.6666374, -7.9502521, -8.6666374, -7.9502521, -0.3872299, 0.3872297
2: -0.8758705, -0.2017304, -0.8758705, -0.2017304, -0.3339977, 0.3339982
3: -4.4505491, -3.5619974, -4.4505491, -3.5619974, -0.4332209, 0.4332209
4: -11.9265947, -11.1367950, -11.9265947, -11.1367950, -0.2893162, 0.2893162
5: -9.1612968, -8.2840586, -9.1612968, -8.2840586, -0.3793964, 0.3793964
6: -11.1298752, -10.2386789, -11.1298752, -10.2386789, -0.4020977, 0.4020977
7: -11.7566013, -10.9271126, -11.7566013, -10.9271126, -0.3698549, 0.3698547
8: 9.1338673, 9.8127308, 9.1338673, 9.8127308, -0.4495106, 0.4495106
9: -5.4181666, -4.7633448, -5.4181666, -4.7633448, -0.2660081, 0.2660081

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 63

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862135, upper bound: 0.1858600
time: 5.76 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865765, upper bound: 0.1858600
time: 3.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.14 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1858603
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1858603
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1865767
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1865764
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1862132, upper bound: 0.1858600
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1865762, upper bound: 0.1858600
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1862135, upper bound: 0.1858600
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.14
Output dim: 8, lower bound: -0.1865765, upper bound: 0.1858600

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.2134795, -4.5120230, -5.2128687, -4.5120339, -0.3043153, 0.3039546
1: -8.6632586, -7.9517698, -8.6634083, -7.9511766, -0.3853188, 0.3849268
2: -0.8753479, -0.2081258, -0.8755248, -0.2075136, -0.3322339, 0.3315575
3: -4.4372272, -3.5631003, -4.4374275, -3.5630407, -0.4303699, 0.4304547
4: -11.9244118, -11.1511154, -11.9245033, -11.1507654, -0.2849720, 0.2848265
5: -9.1416740, -8.2849102, -9.1416779, -8.2856770, -0.3747358, 0.3750563
6: -11.1078110, -10.2407093, -11.1085691, -10.2404785, -0.3978601, 0.3991554
7: -11.7558498, -10.9340658, -11.7561760, -10.9328394, -0.3680892, 0.3670382
8: 9.1386070, 9.8123035, 9.1385603, 9.8123331, -0.4482374, 0.4484053
9: -5.4152455, -4.7837887, -5.4154105, -4.7831988, -0.2599132, 0.2595010

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854971, upper bound: 0.1854973
time: 4.08 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854971, upper bound: 0.1858603
time: 3.83 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.2128682, -4.5121641, -5.2128687, -4.5120316, -0.3040268, 0.3041432
1: -8.6634035, -7.9508061, -8.6634064, -7.9507961, -0.3858886, 0.3848674
2: -0.8755251, -0.2071397, -0.8755257, -0.2071171, -0.3318181, 0.3323631
3: -4.4375510, -3.5630407, -4.4375553, -3.5630407, -0.4304347, 0.4305692
4: -11.9244881, -11.1505394, -11.9245024, -11.1505375, -0.2852712, 0.2848325
5: -9.1415300, -8.2856779, -9.1416817, -8.2856779, -0.3744278, 0.3754797
6: -11.1090612, -10.2404785, -11.1090651, -10.2404785, -0.3993063, 0.3981266
7: -11.7561760, -10.9320650, -11.7561760, -10.9320564, -0.3688111, 0.3674626
8: 9.1385212, 9.8123341, 9.1385126, 9.8123331, -0.4482765, 0.4482317
9: -5.4153919, -4.7828236, -5.4154100, -4.7828226, -0.2604325, 0.2595103

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858601, upper bound: 0.1854969
time: 6.17 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858601, upper bound: 0.1858600
time: 5.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.2134795, -4.5120230, -5.2142587, -4.4942474, -0.3125459, 0.3053536
1: -8.6632586, -7.9517698, -8.6666384, -7.9506321, -0.3859529, 0.3881907
2: -0.8753479, -0.2081258, -0.8758706, -0.2021292, -0.3375564, 0.3319066
3: -4.4372272, -3.5631003, -4.4504204, -3.5619974, -0.4314284, 0.4401348
4: -11.9244118, -11.1511154, -11.9265928, -11.1370201, -0.2936549, 0.2872062
5: -9.1416740, -8.2849102, -9.1612911, -8.2840586, -0.3763757, 0.3829389
6: -11.1078110, -10.2407093, -11.1293783, -10.2386789, -0.3996902, 0.4113319
7: -11.7558498, -10.9340658, -11.7566013, -10.9278946, -0.3730206, 0.3674731
8: 9.1386070, 9.8123035, 9.1339140, 9.8127308, -0.4486365, 0.4530911
9: -5.4152455, -4.7837887, -5.4181633, -4.7637215, -0.2668184, 0.2628024

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1862134
time: 3.90 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1865764
time: 4.53 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.2128682, -4.5121641, -5.2142577, -4.4942460, -0.3130453, 0.3055425
1: -8.6634035, -7.9508061, -8.6666374, -7.9502521, -0.3865232, 0.3881311
2: -0.8755251, -0.2071397, -0.8758705, -0.2017304, -0.3371081, 0.3327117
3: -4.4375510, -3.5630407, -4.4505491, -3.5619974, -0.4314923, 0.4404027
4: -11.9244881, -11.1505394, -11.9265947, -11.1367950, -0.2940947, 0.2872126
5: -9.1415300, -8.2856779, -9.1612968, -8.2840586, -0.3760676, 0.3835495
6: -11.1090612, -10.2404785, -11.1298752, -10.2386789, -0.4011374, 0.4123616
7: -11.7561760, -10.9320650, -11.7566013, -10.9271126, -0.3737431, 0.3678980
8: 9.1385212, 9.8123341, 9.1338673, 9.8127308, -0.4486766, 0.4529185
9: -5.4153919, -4.7828236, -5.4181666, -4.7633448, -0.2675653, 0.2628093

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1862133
time: 5.65 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1865761
time: 6.34 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.2148700, -4.4942389, -5.2128687, -4.5120339, -0.3057146, 0.3125464
1: -8.6664925, -7.9512272, -8.6634083, -7.9511766, -0.3885818, 0.3855605
2: -0.8756939, -0.2027399, -0.8755248, -0.2075136, -0.3325830, 0.3368819
3: -4.4502211, -3.5620575, -4.4374275, -3.5630407, -0.4401333, 0.4315133
4: -11.9265022, -11.1373682, -11.9245033, -11.1507654, -0.2873526, 0.2936429
5: -9.1612873, -8.2832890, -9.1416779, -8.2856770, -0.3829319, 0.3766966
6: -11.1286163, -10.2389145, -11.1085691, -10.2404785, -0.4113362, 0.4009862
7: -11.7562771, -10.9291229, -11.7561760, -10.9328394, -0.3685241, 0.3719702
8: 9.1339607, 9.8127022, 9.1385603, 9.8123331, -0.4529243, 0.4488053
9: -5.4180021, -4.7643108, -5.4154105, -4.7831988, -0.2632129, 0.2668025

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862132, upper bound: 0.1854970
time: 6.48 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1862132, upper bound: 0.1858600
time: 4.47 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.2142577, -4.4943800, -5.2128687, -4.5120316, -0.3054256, 0.3125528
1: -8.6666365, -7.9502597, -8.6634064, -7.9507961, -0.3891516, 0.3855011
2: -0.8758708, -0.2017547, -0.8755257, -0.2071171, -0.3321662, 0.3376863
3: -4.4505453, -3.5619974, -4.4375553, -3.5630407, -0.4401364, 0.4316282
4: -11.9265785, -11.1367960, -11.9245024, -11.1505375, -0.2876518, 0.2936604
5: -9.1611443, -8.2840605, -9.1416817, -8.2856779, -0.3829446, 0.3771200
6: -11.1298695, -10.2386808, -11.1090651, -10.2404785, -0.4113724, 0.3999574
7: -11.7566032, -10.9271202, -11.7561760, -10.9320564, -0.3692460, 0.3723948
8: 9.1338730, 9.8127317, 9.1385126, 9.8123331, -0.4529629, 0.4486308
9: -5.4181485, -4.7633471, -5.4154100, -4.7828226, -0.2637339, 0.2668355

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865762, upper bound: 0.1854970
time: 3.92 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865762, upper bound: 0.1858600
time: 3.76 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.2148700, -4.4942389, -5.2142587, -4.4942474, -0.3086283, 0.3082676
1: -8.6664925, -7.9512272, -8.6666384, -7.9506321, -0.3866448, 0.3862524
2: -0.8756939, -0.2027399, -0.8758706, -0.2021292, -0.3344471, 0.3337729
3: -4.4502211, -3.5620575, -4.4504204, -3.5619974, -0.4330306, 0.4331055
4: -11.9265022, -11.1373682, -11.9265928, -11.1370201, -0.2889946, 0.2888496
5: -9.1612873, -8.2832890, -9.1612911, -8.2840586, -0.3786488, 0.3789692
6: -11.1286163, -10.2389145, -11.1293783, -10.2386789, -0.4018307, 0.4031262
7: -11.7562771, -10.9291229, -11.7566013, -10.9278946, -0.3691325, 0.3680816
8: 9.1339607, 9.8127022, 9.1339140, 9.8127308, -0.4495163, 0.4496832
9: -5.4180021, -4.7643108, -5.4181633, -4.7637215, -0.2654550, 0.2650447

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1854970
time: 4.41 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1858687
time: 3.86 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.2142577, -4.4943800, -5.2142577, -4.4942460, -0.3083050, 0.3084559
1: -8.6666365, -7.9502597, -8.6666374, -7.9502521, -0.3872156, 0.3861933
2: -0.8758708, -0.2017547, -0.8758705, -0.2017304, -0.3339982, 0.3345776
3: -4.4505453, -3.5619974, -4.4505491, -3.5619974, -0.4330969, 0.4332209
4: -11.9265785, -11.1367960, -11.9265947, -11.1367950, -0.2892945, 0.2888558
5: -9.1611443, -8.2840605, -9.1612968, -8.2840586, -0.3783417, 0.3793926
6: -11.1298695, -10.2386808, -11.1298752, -10.2386789, -0.4032769, 0.4020965
7: -11.7566032, -10.9271202, -11.7566013, -10.9271126, -0.3698545, 0.3685060
8: 9.1338730, 9.8127317, 9.1338673, 9.8127308, -0.4495549, 0.4495101
9: -5.4181485, -4.7633471, -5.4181666, -4.7633448, -0.2659764, 0.2650516

Time for backsubstitution: 20.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 63

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1855054
time: 7.36 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1858687
time: 5.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.72 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1854971, upper bound: 0.1854973
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1854971, upper bound: 0.1858603
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1858601, upper bound: 0.1854969
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1858601, upper bound: 0.1858600
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1862134
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1865764
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1862133
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1858598, upper bound: 0.1865761
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1862132, upper bound: 0.1854970
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1862132, upper bound: 0.1858600
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1865762, upper bound: 0.1854970
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1865762, upper bound: 0.1858600
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1854970
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1854968, upper bound: 0.1858687
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1855054
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.72
Output dim: 8, lower bound: -0.1865833, upper bound: 0.1858687

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.2134795, -4.5120230, -5.2134795, -4.5120230, -0.3043120, 0.3043120
1: -8.6632586, -7.9517698, -8.6632586, -7.9517698, -0.3847537, 0.3847535
2: -0.8753479, -0.2081258, -0.8753479, -0.2081258, -0.3322344, 0.3322344
3: -4.4372272, -3.5631003, -4.4372272, -3.5631003, -0.4303842, 0.4303842
4: -11.9244118, -11.1511154, -11.9244118, -11.1511154, -0.2847295, 0.2847295
5: -9.1416740, -8.2849102, -9.1416740, -8.2849102, -0.3746071, 0.3746071
6: -11.1078110, -10.2407093, -11.1078110, -10.2407093, -0.3991585, 0.3991587
7: -11.7558498, -10.9340658, -11.7558498, -10.9340658, -0.3670931, 0.3670931
8: 9.1386070, 9.8123035, 9.1386070, 9.8123035, -0.4484124, 0.4484124
9: -5.4152455, -4.7837887, -5.4152455, -4.7837887, -0.2593257, 0.2593257

Time for backsubstitution: 20.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.2134795, -4.5120230, -5.2128682, -4.5121641, -0.3043649, 0.3039546
1: -8.6632586, -7.9517698, -8.6634035, -7.9508061, -0.3857183, 0.3849134
2: -0.8753479, -0.2081258, -0.8755251, -0.2071397, -0.3324714, 0.3315580
3: -4.4372272, -3.5631003, -4.4375510, -3.5630407, -0.4303699, 0.4305787
4: -11.9244118, -11.1511154, -11.9244881, -11.1505394, -0.2851942, 0.2848048
5: -9.1416740, -8.2849102, -9.1415300, -8.2856779, -0.3747325, 0.3753443
6: -11.1078110, -10.2407093, -11.1090612, -10.2404785, -0.3978586, 0.3994174
7: -11.7558498, -10.9340658, -11.7561760, -10.9320650, -0.3688555, 0.3670385
8: 9.1386070, 9.8123035, 9.1385212, 9.8123341, -0.4482374, 0.4483738
9: -5.4152455, -4.7837887, -5.4153919, -4.7828236, -0.2602861, 0.2594700

Time for backsubstitution: 19.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.2128682, -4.5121641, -5.2134795, -4.5120230, -0.3039546, 0.3043649
1: -8.6634035, -7.9508061, -8.6632586, -7.9517698, -0.3849134, 0.3857181
2: -0.8755251, -0.2071397, -0.8753479, -0.2081258, -0.3315578, 0.3324716
3: -4.4375510, -3.5630407, -4.4372272, -3.5631003, -0.4305787, 0.4303699
4: -11.9244881, -11.1505394, -11.9244118, -11.1511154, -0.2848048, 0.2851942
5: -9.1415300, -8.2856779, -9.1416740, -8.2849102, -0.3753438, 0.3747325
6: -11.1090612, -10.2404785, -11.1078110, -10.2407093, -0.3994174, 0.3978586
7: -11.7561760, -10.9320650, -11.7558498, -10.9340658, -0.3670387, 0.3688555
8: 9.1385212, 9.8123341, 9.1386070, 9.8123035, -0.4483738, 0.4482374
9: -5.4153919, -4.7828236, -5.4152455, -4.7837887, -0.2594700, 0.2602861

Time for backsubstitution: 19.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.2128682, -4.5121641, -5.2128682, -4.5121641, -0.3041432, 0.3041432
1: -8.6634035, -7.9508061, -8.6634035, -7.9508061, -0.3848534, 0.3848534
2: -0.8755251, -0.2071397, -0.8755251, -0.2071397, -0.3323629, 0.3323629
3: -4.4375510, -3.5630407, -4.4375510, -3.5630407, -0.4304342, 0.4304342
4: -11.9244881, -11.1505394, -11.9244881, -11.1505394, -0.2848103, 0.2848103
5: -9.1415300, -8.2856779, -9.1415300, -8.2856779, -0.3744235, 0.3744235
6: -11.1090612, -10.2404785, -11.1090612, -10.2404785, -0.3993053, 0.3993053
7: -11.7561760, -10.9320650, -11.7561760, -10.9320650, -0.3674626, 0.3674626
8: 9.1385212, 9.8123341, 9.1385212, 9.8123341, -0.4482760, 0.4482760
9: -5.4153919, -4.7828236, -5.4153919, -4.7828236, -0.2594786, 0.2594783

Time for backsubstitution: 20.86 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.63 + 554.40 = 612.03 seconds
