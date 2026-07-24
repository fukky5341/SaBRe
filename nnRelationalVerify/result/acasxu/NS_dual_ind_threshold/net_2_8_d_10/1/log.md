## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 24053.216940845126


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844)
1: (-1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172)
2: (-2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422)
3: (-2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164)
4: (-2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.83 + 2.23 = 5.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24054.9007839, upper bound: 24054.9007839

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.9002765, upper bound: 24054.9001356
time: 0.82 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.9001334, upper bound: 24054.9001334
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.87 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 0, lower bound: -24054.9002765, upper bound: 24054.9001356
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 0, lower bound: -24054.9001334, upper bound: 24054.9001334

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12099.6015625, 12651.8359375, -12389.2724609, 13009.6582031, -25109.2558594, 25041.1074219
1: -1284.8458252, 935.3571777, -1317.2768555, 956.6473999, -2241.4931641, 2252.6340332
2: -2099.4689941, 2403.5898438, -2153.4765625, 2469.4475098, -4568.9160156, 4557.0664062
3: -2431.2653809, 1512.8117676, -2493.2871094, 1552.2846680, -3983.5500488, 4006.0988770
4: -1843.2506104, 1928.5170898, -1885.3726807, 1983.2220459, -3826.4719238, 3813.8896484

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8983920, upper bound: 24054.8984466
time: 0.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.9002765, upper bound: 24054.9001356
time: 0.80 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13427.4619141, 14183.2431641, -13492.4755859, 14247.3291016, -27674.7910156, 27675.7187500
1: -1434.8791504, 1032.9298096, -1441.5109863, 1037.9351807, -2472.8139648, 2474.4404297
2: -2337.8833008, 2690.2385254, -2349.0129395, 2702.6181641, -5040.5014648, 5039.2514648
3: -2711.6965332, 1688.8914795, -2724.4567871, 1696.7218018, -4408.4184570, 4413.3481445
4: -2047.1713867, 2160.6127930, -2056.9333496, 2170.4450684, -4217.6162109, 4217.5458984

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8986383, upper bound: 24054.8988695
time: 0.91 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.9001334, upper bound: 24054.9001334
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.47 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -24054.8983920, upper bound: 24054.8984466
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -24054.9002765, upper bound: 24054.9001356
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -24054.8986383, upper bound: 24054.8988695
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -24054.9001334, upper bound: 24054.9001334

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -11745.6474609, 12258.1689453, -12084.1054688, 12600.7587891, -24346.4042969, 24342.2734375
1: -1245.1342773, 908.2521362, -1279.7393799, 931.1744385, -2176.3081055, 2187.9909668
2: -2039.6671143, 2333.9304199, -2098.0141602, 2402.4409180, -4442.1079102, 4431.9443359
3: -2364.5324707, 1467.7005615, -2435.8286133, 1507.9829102, -3872.5153809, 3903.5292969
4: -1791.6011963, 1871.9339600, -1845.2928467, 1926.6715088, -3718.2719727, 3717.2268066

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8769296, upper bound: 24054.8782710
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8771087, upper bound: 24054.8780517
time: 0.83 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -11848.5976562, 12396.2353516, -12000.3281250, 12568.3906250, -24416.9882812, 24396.5625000
1: -1259.0262451, 915.6897583, -1275.9628906, 927.2084961, -2186.2343750, 2191.6525879
2: -2057.6098633, 2355.4230957, -2087.2919922, 2387.7072754, -4445.3173828, 4442.7148438
3: -2385.1022949, 1482.0404053, -2420.5625000, 1502.0625000, -3887.1647949, 3902.6030273
4: -1807.7812500, 1889.3383789, -1831.5281982, 1914.8355713, -3722.6166992, 3720.8664551

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8817568, upper bound: 24054.8813931
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811734, upper bound: 24054.8805547
time: 0.73 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13256.8359375, 14010.1181641, -12997.2226562, 13777.9501953, -27034.7851562, 27007.3398438
1: -1417.3360596, 1019.8624878, -1392.2673340, 999.1738281, -2416.5095215, 2412.1298828
2: -2308.3303223, 2657.8542480, -2261.0988770, 2609.7309570, -4918.0615234, 4918.9521484
3: -2678.3688965, 1668.0572510, -2616.7243652, 1637.6279297, -4315.9965820, 4284.7812500
4: -2021.6140137, 2134.4265137, -1976.8948975, 2097.9992676, -4119.6132812, 4111.3212891

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8751157, upper bound: 24054.8780295
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8741991, upper bound: 24054.8752552
time: 0.94 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12911.6279297, 13582.1738281, -12713.2978516, 13338.2841797, -26249.9121094, 26295.4726562
1: -1377.0722656, 993.7559204, -1354.1330566, 978.7909546, -2355.8632812, 2347.8884277
2: -2250.2355957, 2582.7216797, -2216.6623535, 2540.0532227, -4790.2885742, 4799.3837891
3: -2614.9599609, 1621.3216553, -2578.5395508, 1594.5876465, -4209.5478516, 4199.8608398
4: -1974.3887939, 2071.0930176, -1947.1607666, 2034.9520264, -4009.3408203, 4018.2539062

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8790930, upper bound: 24054.8808553
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8775131, upper bound: 24054.8775131
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.31 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8769296, upper bound: 24054.8782710
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8771087, upper bound: 24054.8780517
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8817568, upper bound: 24054.8813931
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8811734, upper bound: 24054.8805547
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8751157, upper bound: 24054.8780295
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8741991, upper bound: 24054.8752552
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8790930, upper bound: 24054.8808553
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 0, lower bound: -24054.8775131, upper bound: 24054.8775131

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11425.6064453, 11947.4931641, -11860.7177734, 12397.5146484, -23823.1171875, 23808.2031250
1: -1212.7934570, 882.6789551, -1257.9926758, 913.5422974, -2126.3354492, 2140.6711426
2: -1986.1142578, 2272.4904785, -2059.8796387, 2361.0705566, -4347.1845703, 4332.3701172
3: -2302.1113281, 1429.2023926, -2391.7807617, 1482.1046143, -3784.2158203, 3820.9831543
4: -1742.1834717, 1823.9984131, -1810.9451904, 1894.0695801, -3636.2529297, 3634.9436035

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8766289, upper bound: 24054.8774070
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8766289, upper bound: 24054.8777692
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12222.7421875, 12748.4453125, -11598.5703125, 12089.5126953, -24312.2539062, 24347.0156250
1: -1295.0737305, 945.1024170, -1228.1387939, 893.6082764, -2188.6818848, 2173.2412109
2: -2123.6774902, 2424.5695801, -2015.1872559, 2306.9057617, -4430.5830078, 4439.7563477
3: -2460.5205078, 1527.8244629, -2341.2023926, 1447.4526367, -3907.9731445, 3869.0268555
4: -1863.9250488, 1945.0404053, -1772.9979248, 1849.7983398, -3713.7233887, 3718.0378418

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8776159
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8780517
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11449.8964844, 11993.6025391, -11744.6572266, 12300.1513672, -23750.0449219, 23738.2558594
1: -1217.5927734, 884.6337891, -1248.4268799, 906.8088989, -2124.4016113, 2133.0605469
2: -1989.9461670, 2277.3129883, -2043.9273682, 2336.1018066, -4326.0478516, 4321.2402344
3: -2306.0737305, 1433.3581543, -2370.9963379, 1469.9138184, -3775.9875488, 3804.3544922
4: -1746.4210205, 1827.4879150, -1793.8876953, 1873.6835938, -3620.1044922, 3621.3754883

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8817568, upper bound: 24054.8813931
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8817568, upper bound: 24054.8813931
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12276.4091797, 12834.3662109, -11545.0537109, 12110.3027344, -24386.7109375, 24379.4199219
1: -1303.8654785, 948.5476074, -1229.2015381, 892.1261597, -2195.9912109, 2177.7487793
2: -2133.6386719, 2436.3898926, -2009.3509521, 2300.5512695, -4434.1894531, 4445.7402344
3: -2471.9538574, 1536.0144043, -2331.0244141, 1446.9954834, -3918.9492188, 3867.0388184
4: -1873.1081543, 1954.6669922, -1762.6685791, 1844.9221191, -3718.0295410, 3717.3354492

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811734, upper bound: 24054.8805547
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811734, upper bound: 24054.8805547
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13001.4853516, 13754.7714844, -12814.1074219, 13596.0468750, -26597.5312500, 26568.8789062
1: -1390.9018555, 1000.1868286, -1373.5634766, 984.9568481, -2375.8586426, 2373.7502441
2: -2264.0429688, 2608.8603516, -2229.7814941, 2574.9084473, -4838.9511719, 4838.6411133
3: -2627.2041016, 1636.9444580, -2581.1145020, 1615.4645996, -4242.6679688, 4218.0585938
4: -1982.4926758, 2095.3320312, -1949.4295654, 2070.1894531, -4052.6813965, 4044.7612305

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8426429, upper bound: 24054.8530856
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8701455, upper bound: 24054.8712208
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13464.5498047, 14202.7236328, -12438.6962891, 13203.8066406, -26668.3554688, 26641.4199219
1: -1438.3293457, 1034.9580078, -1333.6246338, 955.6962280, -2394.0256348, 2368.5820312
2: -2347.9577637, 2694.1501465, -2165.5014648, 2500.2231445, -4848.1791992, 4859.6513672
3: -2725.2712402, 1692.7750244, -2506.2644043, 1568.8149414, -4294.0859375, 4199.0390625
4: -2056.3579102, 2164.0979004, -1892.1958008, 2010.3676758, -4066.7253418, 4056.2937012

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8424467, upper bound: 24054.8525078
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8692135, upper bound: 24054.8690225
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12635.3037109, 13306.0634766, -12528.3818359, 13153.5517578, -25788.8554688, 25834.4453125
1: -1348.5344238, 972.4018555, -1335.1075439, 964.5043335, -2313.0388184, 2307.5092773
2: -2202.4560547, 2529.7504883, -2184.8508301, 2504.5563965, -4707.0126953, 4714.6005859
3: -2560.0441895, 1587.6464844, -2542.1630859, 1572.0576172, -4132.1015625, 4129.8095703
4: -1932.3594971, 2028.8752441, -1919.3094482, 2006.6673584, -3939.0266113, 3948.1843262

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8790930, upper bound: 24054.8808553
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8790930, upper bound: 24054.8808553
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13097.1796875, 13757.9589844, -12158.6035156, 12765.9121094, -25863.0917969, 25916.5625000
1: -1396.0568848, 1007.1751709, -1295.7761230, 935.4475098, -2331.5043945, 2302.9509277
2: -2285.9311523, 2615.1916504, -2122.2543945, 2431.0942383, -4717.0253906, 4737.4458008
3: -2656.9150391, 1643.7666016, -2470.0878906, 1525.9616699, -4182.8759766, 4113.8544922
4: -2005.4670410, 2097.7312012, -1863.8162842, 1947.6452637, -3953.1115723, 3961.5473633

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8775131, upper bound: 24054.8775131
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8775131, upper bound: 24054.8775131
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.03 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8766289, upper bound: 24054.8774070
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8766289, upper bound: 24054.8777692
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8776159
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8780517
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8817568, upper bound: 24054.8813931
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8817568, upper bound: 24054.8813931
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8811734, upper bound: 24054.8805547
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8811734, upper bound: 24054.8805547
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8426429, upper bound: 24054.8530856
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8701455, upper bound: 24054.8712208
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8424467, upper bound: 24054.8525078
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8692135, upper bound: 24054.8690225
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8790930, upper bound: 24054.8808553
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8790930, upper bound: 24054.8808553
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8775131, upper bound: 24054.8775131
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 0, lower bound: -24054.8775131, upper bound: 24054.8775131

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11425.6064453, 11947.4931641, -11717.5439453, 12266.7177734, -23692.3242188, 23665.0351562
1: -1212.7934570, 882.6789551, -1244.0078125, 902.2191162, -2115.0126953, 2126.6867676
2: -1986.1142578, 2272.4904785, -2035.4305420, 2334.5126953, -4320.6269531, 4307.9208984
3: -2302.1113281, 1429.2023926, -2363.4667969, 1465.5134277, -3767.6247559, 3792.6691895
4: -1742.1834717, 1823.9984131, -1788.8676758, 1873.1342773, -3615.3173828, 3612.8657227

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8670596, upper bound: 24054.8727777
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11425.6064453, 11947.4931641, -12082.7568359, 12600.1152344, -24025.7226562, 24030.2500000
1: -1212.7934570, 882.6789551, -1279.9669189, 931.0083618, -2143.8017578, 2162.6457520
2: -1986.1142578, 2272.4904785, -2100.0566406, 2401.4350586, -4387.5493164, 4372.5468750
3: -2302.1113281, 1429.2023926, -2437.7692871, 1508.7702637, -3810.8815918, 3866.9716797
4: -1742.1834717, 1823.9984131, -1846.9101562, 1926.5966797, -3668.7800293, 3670.9086914

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8670596, upper bound: 24054.8727777
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12222.7421875, 12748.4453125, -11717.5439453, 12266.7177734, -24489.4609375, 24465.9882812
1: -1295.0737305, 945.1024170, -1244.0078125, 902.2191162, -2197.2929688, 2189.1098633
2: -2123.6774902, 2424.5695801, -2035.4305420, 2334.5126953, -4458.1894531, 4459.9995117
3: -2460.5205078, 1527.8244629, -2363.4667969, 1465.5134277, -3926.0339355, 3891.2912598
4: -1863.9250488, 1945.0404053, -1788.8676758, 1873.1342773, -3737.0590820, 3733.9077148

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8776159
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8776134
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12222.7421875, 12748.4453125, -12082.7568359, 12600.1152344, -24822.8574219, 24831.2031250
1: -1295.0737305, 945.1024170, -1279.9669189, 931.0083618, -2226.0820312, 2225.0690918
2: -2123.6774902, 2424.5695801, -2100.0566406, 2401.4350586, -4525.1123047, 4524.6254883
3: -2460.5205078, 1527.8244629, -2437.7692871, 1508.7702637, -3969.2907715, 3965.5937500
4: -1863.9250488, 1945.0404053, -1846.9101562, 1926.5966797, -3790.5217285, 3791.9504395

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8780491
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8778593
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11449.8964844, 11993.6025391, -11496.2421875, 12041.9277344, -23491.8222656, 23489.8437500
1: -1217.5927734, 884.6337891, -1222.6512451, 888.2644043, -2105.8571777, 2107.2851562
2: -1989.9461670, 2277.3129883, -1997.8511963, 2287.2695312, -4277.2158203, 4275.1640625
3: -2306.0737305, 1433.3581543, -2316.2639160, 1439.0882568, -3745.1621094, 3749.6220703
4: -1746.4210205, 1827.4879150, -1754.4815674, 1835.1329346, -3581.5539551, 3581.9694824

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811746
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11449.8964844, 11993.6025391, -12448.7871094, 13074.1025391, -24523.9980469, 24442.3886719
1: -1217.5927734, 884.6337891, -1326.8956299, 958.3775635, -2175.9702148, 2211.5292969
2: -1989.9461670, 2277.3129883, -2171.0913086, 2489.3347168, -4479.2807617, 4448.4042969
3: -2306.0737305, 1433.3581543, -2526.2741699, 1562.4166260, -3868.4902344, 3959.6323242
4: -1746.4210205, 1827.4879150, -1907.1759033, 1994.5274658, -3740.9479980, 3734.6638184

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811746
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12276.4091797, 12834.3662109, -11275.9521484, 11813.7822266, -24090.1894531, 24110.3144531
1: -1303.8654785, 948.5476074, -1199.6008301, 871.1846924, -2175.0495605, 2148.1484375
2: -2133.6386719, 2436.3898926, -1959.6826172, 2244.8310547, -4378.4697266, 4396.0717773
3: -2471.9538574, 1536.0144043, -2272.7995605, 1411.7568359, -3883.7104492, 3808.8139648
4: -1873.1081543, 1954.6669922, -1721.8457031, 1800.5520020, -3673.6601562, 3676.5126953

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8594325, upper bound: 24054.8571153
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12276.4091797, 12834.3662109, -12062.0986328, 12670.5449219, -24946.9531250, 24896.4628906
1: -1303.8654785, 948.5476074, -1285.9514160, 927.9892578, -2231.8542480, 2234.4985352
2: -2133.6386719, 2436.3898926, -2105.8781738, 2412.8576660, -4546.4960938, 4542.2670898
3: -2471.9538574, 1536.0144043, -2451.3608398, 1514.3483887, -3986.3017578, 3987.3752441
4: -1873.1081543, 1954.6669922, -1849.4587402, 1933.0429688, -3806.1511230, 3804.1257324

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8594325, upper bound: 24054.8571153
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
time: 3.42 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12884.1591797, 13636.0791016, -12647.7050781, 13432.7382812, -26316.8984375, 26283.7851562
1: -1378.7628174, 991.1181030, -1356.6655273, 971.9248657, -2350.6877441, 2347.7836914
2: -2243.9228516, 2586.6450195, -2202.4672852, 2544.9704590, -4788.8920898, 4789.1123047
3: -2604.3527832, 1622.5080566, -2550.6660156, 1595.0233154, -4199.3759766, 4173.1738281
4: -1964.8715820, 2077.5229492, -1926.1687012, 2046.4304199, -4011.3020020, 4003.6911621

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8515250
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8530856
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12682.3896484, 13384.6406250, -12316.4931641, 13004.1650391, -25686.5546875, 25701.1328125
1: -1354.8499756, 975.6398926, -1316.0555420, 946.8024292, -2301.6523438, 2291.6953125
2: -2210.5048828, 2541.8032227, -2145.6040039, 2468.3325195, -4678.8369141, 4687.4072266
3: -2566.8156738, 1594.5047607, -2485.8327637, 1548.1496582, -4114.9653320, 4080.3371582
4: -1937.1949463, 2040.8549805, -1878.3775635, 1983.2563477, -3920.4511719, 3919.2319336

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8657268, upper bound: 24054.8686340
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8657268, upper bound: 24054.8712208
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13358.1240234, 14089.2802734, -12328.6689453, 13095.1982422, -26453.3222656, 26417.9492188
1: -1426.8507080, 1026.6607666, -1322.5061035, 946.9402466, -2373.7910156, 2349.1669922
2: -2329.7028809, 2673.0778809, -2148.4428711, 2480.6491699, -4810.3520508, 4821.5195312
3: -2704.4772949, 1679.1909180, -2488.1520996, 1555.1298828, -4259.6064453, 4167.3417969
4: -2040.4766846, 2147.2795410, -1878.4566650, 1994.7399902, -4035.2167969, 4025.7363281

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8505057
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8525078
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13173.4150391, 13871.1513672, -11994.7978516, 12680.6806641, -25854.0957031, 25865.9492188
1: -1405.9544678, 1012.5107422, -1282.6116943, 921.5219116, -2327.4763184, 2295.1225586
2: -2299.4069824, 2633.9338379, -2090.8935547, 2405.7785645, -4705.1855469, 4724.8266602
3: -2670.7048340, 1654.6297607, -2422.1416016, 1509.1359863, -4179.8403320, 4076.7712402
4: -2015.3183594, 2115.2534180, -1829.3028564, 1933.4008789, -3948.7192383, 3944.5554199

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8656209, upper bound: 24054.8656209
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8656209, upper bound: 24054.8690225
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12635.3037109, 13306.0634766, -11496.2421875, 12041.9277344, -24677.2285156, 24802.3046875
1: -1348.5344238, 972.4018555, -1222.6512451, 888.2644043, -2236.7988281, 2195.0532227
2: -2202.4560547, 2529.7504883, -1997.8511963, 2287.2695312, -4489.7255859, 4527.6015625
3: -2560.0441895, 1587.6464844, -2316.2639160, 1439.0882568, -3999.1323242, 3903.9104004
4: -1932.3594971, 2028.8752441, -1754.4815674, 1835.1329346, -3767.4924316, 3783.3564453

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782689, upper bound: 24054.8808000
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8778099, upper bound: 24054.8794548
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12635.3037109, 13306.0634766, -12448.7871094, 13074.1025391, -25709.4023438, 25754.8476562
1: -1348.5344238, 972.4018555, -1326.8956299, 958.3775635, -2306.9118652, 2299.2973633
2: -2202.4560547, 2529.7504883, -2171.0913086, 2489.3347168, -4691.7910156, 4700.8417969
3: -2560.0441895, 1587.6464844, -2526.2741699, 1562.4166260, -4122.4609375, 4113.9208984
4: -1932.3594971, 2028.8752441, -1907.1759033, 1994.5274658, -3926.8869629, 3936.0502930

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782689, upper bound: 24054.8808000
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8778099, upper bound: 24054.8794548
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -13097.1796875, 13757.9589844, -11275.9521484, 11813.7822266, -24910.9609375, 25033.9101562
1: -1396.0568848, 1007.1751709, -1199.6008301, 871.1846924, -2267.2416992, 2206.7756348
2: -2285.9311523, 2615.1916504, -1959.6826172, 2244.8310547, -4530.7617188, 4574.8740234
3: -2656.9150391, 1643.7666016, -2272.7995605, 1411.7568359, -4068.6716309, 3916.5659180
4: -2005.4670410, 2097.7312012, -1721.8457031, 1800.5520020, -3806.0183105, 3819.5769043

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8592150, upper bound: 24054.8549413
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -13097.1796875, 13757.9589844, -12062.0986328, 12670.5449219, -25767.7246094, 25820.0585938
1: -1396.0568848, 1007.1751709, -1285.9514160, 927.9892578, -2324.0461426, 2293.1254883
2: -2285.9311523, 2615.1916504, -2105.8781738, 2412.8576660, -4698.7880859, 4721.0688477
3: -2656.9150391, 1643.7666016, -2451.3608398, 1514.3483887, -4171.2622070, 4095.1274414
4: -2005.4670410, 2097.7312012, -1849.4587402, 1933.0429688, -3938.5095215, 3947.1899414

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8592150, upper bound: 24054.8549413
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.25 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8670596, upper bound: 24054.8727777
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8670596, upper bound: 24054.8727777
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8776159
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8776134
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8780491
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8767160, upper bound: 24054.8778593
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811746
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811746
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8594325, upper bound: 24054.8571153
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8594325, upper bound: 24054.8571153
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8515250
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8530856
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8657268, upper bound: 24054.8686340
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8657268, upper bound: 24054.8712208
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8505057
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8525078
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8656209, upper bound: 24054.8656209
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8656209, upper bound: 24054.8690225
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8782689, upper bound: 24054.8808000
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8778099, upper bound: 24054.8794548
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8782689, upper bound: 24054.8808000
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8778099, upper bound: 24054.8794548
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8592150, upper bound: 24054.8549413
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8592150, upper bound: 24054.8549413
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.25
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11403.2236328, 11863.7919922, -11545.4511719, 12074.3095703, -23477.5332031, 23409.2421875
1: -1205.8679199, 878.2639160, -1224.9617920, 888.9399414, -2094.8078613, 2103.2253418
2: -1987.3432617, 2257.0263672, -2006.4202881, 2298.9978027, -4286.3408203, 4263.4467773
3: -2309.8676758, 1421.4077148, -2330.6625977, 1443.1588135, -3753.0263672, 3752.0700684
4: -1747.7548828, 1810.4537354, -1764.1928711, 1844.3719482, -3592.1269531, 3574.6464844

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8772841, upper bound: 24054.8734013
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8790035, upper bound: 24054.8751033
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11326.4472656, 11848.8193359, -11620.2939453, 12168.9433594, -23495.3867188, 23469.1132812
1: -1202.5164795, 875.0122070, -1233.8883057, 894.6762695, -2097.1928711, 2108.9001465
2: -1968.9570312, 2253.5725098, -2018.3843994, 2315.7607422, -4284.7172852, 4271.9570312
3: -2282.3962402, 1417.0601807, -2343.5729980, 1453.5648193, -3735.9609375, 3760.6328125
4: -1726.9194336, 1808.9422607, -1773.7503662, 1858.1994629, -3585.1184082, 3582.6926270

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8761779, upper bound: 24054.8731784
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8772804, upper bound: 24054.8748116
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11403.2236328, 11863.7919922, -11933.3935547, 12441.5517578, -23844.7753906, 23797.1855469
1: -1205.8679199, 878.2639160, -1264.1092529, 919.2896729, -2125.1577148, 2142.3730469
2: -1987.3432617, 2257.0263672, -2074.8701172, 2371.7199707, -4359.0634766, 4331.8964844
3: -2309.8676758, 1421.4077148, -2409.7553711, 1490.0941162, -3799.9614258, 3831.1630859
4: -1747.7548828, 1810.4537354, -1825.4462891, 1902.4003906, -3650.1550293, 3635.8999023

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11326.4472656, 11848.8193359, -11959.2431641, 12469.1113281, -23795.5566406, 23808.0625000
1: -1202.5164795, 875.0122070, -1266.5233154, 921.6007690, -2124.1171875, 2141.5356445
2: -1968.9570312, 2253.5725098, -2078.5671387, 2376.6491699, -4345.6059570, 4332.1396484
3: -2282.3962402, 1417.0601807, -2412.3837891, 1492.9779053, -3775.3740234, 3829.4438477
4: -1726.9194336, 1808.9422607, -1827.9099121, 1907.0151367, -3633.9345703, 3636.8518066

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12297.5058594, 12870.1416016, -11419.6044922, 11966.8574219, -24264.3613281, 24289.7460938
1: -1304.6456299, 950.1627197, -1213.2875977, 878.9739990, -2183.6196289, 2163.4501953
2: -2134.7636719, 2442.1884766, -1983.7668457, 2276.8146973, -4411.5781250, 4425.9550781
3: -2467.6867676, 1539.9093018, -2303.1528320, 1429.0793457, -3896.7661133, 3843.0617676
4: -1868.6373291, 1961.4570312, -1742.8051758, 1827.3602295, -3695.9975586, 3704.2622070

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8809156, upper bound: 24054.8784381
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8797919, upper bound: 24054.8785250
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12078.3056641, 12605.3798828, -11567.3750000, 12128.0898438, -24206.3925781, 24172.7519531
1: -1280.0688477, 933.6445312, -1228.7811279, 890.6915283, -2170.7602539, 2162.4257812
2: -2099.6635742, 2395.6269531, -2008.8599854, 2305.8730469, -4405.5366211, 4404.4868164
3: -2432.3964844, 1510.0540771, -2331.2399902, 1447.7697754, -3880.1660156, 3841.2937012
4: -1841.8936768, 1922.7069092, -1764.3273926, 1850.7109375, -3692.6044922, 3687.0334473

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8809156, upper bound: 24054.8784381
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8797919, upper bound: 24054.8785250
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12297.5058594, 12870.1416016, -11781.5634766, 12296.7265625, -24594.2304688, 24651.7031250
1: -1304.6456299, 950.1627197, -1248.6817627, 907.5716553, -2212.2170410, 2198.8439941
2: -2134.7636719, 2442.1884766, -2047.2611084, 2342.9174805, -4477.6811523, 4489.4492188
3: -2467.6867676, 1539.9093018, -2375.6274414, 1471.6904297, -3939.3771973, 3915.5366211
4: -1868.6373291, 1961.4570312, -1799.5085449, 1880.1258545, -3748.7622070, 3760.9655762

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8765868
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8778593
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12078.3056641, 12605.3798828, -11963.7060547, 12490.3837891, -24568.6855469, 24569.0820312
1: -1280.0688477, 933.6445312, -1267.9639893, 921.8718872, -2201.9406738, 2201.6083984
2: -2099.6635742, 2395.6269531, -2078.9545898, 2378.6606445, -4478.3242188, 4474.5815430
3: -2432.3964844, 1510.0540771, -2412.2026367, 1494.7156982, -3927.1123047, 3922.2568359
4: -1841.8936768, 1922.7069092, -1827.3879395, 1908.8154297, -3750.7089844, 3750.0939941

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8765868
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8778593
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11222.9550781, 11750.8144531, -11348.4277344, 11886.2050781, -23109.1582031, 23099.2402344
1: -1192.8704834, 866.6017456, -1206.7696533, 876.6854858, -2069.5559082, 2073.3713379
2: -1951.5623779, 2230.5021973, -1971.9752197, 2257.0834961, -4208.6459961, 4202.4770508
3: -2262.1411133, 1404.3427734, -2285.8261719, 1420.5432129, -3682.6843262, 3690.1689453
4: -1712.4578857, 1790.1953125, -1730.9201660, 1811.2072754, -3523.6650391, 3521.1154785

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8631228, upper bound: 24054.8770984
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8835610, upper bound: 24054.8854778
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11893.7626953, 12417.8007812, -11244.6523438, 11773.3369141, -23667.0996094, 23662.4531250
1: -1261.8587646, 918.0317993, -1195.3750000, 868.5241699, -2130.3828125, 2113.4067383
2: -2067.5588379, 2356.4797363, -1953.4661865, 2235.8156738, -4303.3740234, 4309.9458008
3: -2394.6762695, 1486.4262695, -2264.3249512, 1407.3770752, -3802.0532227, 3750.7507324
4: -1813.6475830, 1891.5272217, -1715.1188965, 1793.9283447, -3607.5759277, 3606.6457520

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8630076, upper bound: 24054.8763662
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8836209, upper bound: 24054.8848433
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11222.9550781, 11750.8144531, -12344.5390625, 12968.4492188, -24191.4023438, 24095.3535156
1: -1192.8704834, 866.6017456, -1315.9667969, 950.3051147, -2143.1755371, 2182.5681152
2: -1951.5623779, 2230.5021973, -2152.8566895, 2469.0915527, -4420.6538086, 4383.3588867
3: -2262.1411133, 1404.3427734, -2505.0639648, 1549.6195068, -3811.7604980, 3909.4067383
4: -1712.4578857, 1790.1953125, -1891.0175781, 1978.4122314, -3690.8698730, 3681.2128906

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11893.7626953, 12417.8007812, -12160.5615234, 12767.3662109, -24661.1289062, 24578.3632812
1: -1261.8587646, 918.0317993, -1295.7279053, 935.6334839, -2197.4921875, 2213.7597656
2: -2067.5588379, 2356.4797363, -2120.1376953, 2430.2546387, -4497.8129883, 4476.6171875
3: -2394.6762695, 1486.4262695, -2466.2485352, 1526.1203613, -3920.7966309, 3952.6745605
4: -1813.6475830, 1891.5272217, -1861.3455811, 1947.3927002, -3761.0400391, 3752.8728027

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811353
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811746
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12136.8769531, 12689.2626953, -11211.5410156, 11747.8779297, -23884.7500000, 23900.8007812
1: -1289.0649414, 937.8256836, -1192.8001709, 866.2496338, -2155.3144531, 2130.6259766
2: -2109.0053711, 2408.5756836, -1948.4017334, 2232.1435547, -4341.1484375, 4356.9775391
3: -2443.0090332, 1518.7714844, -2259.6298828, 1403.7999268, -3846.8090820, 3778.4013672
4: -1850.8012695, 1932.5344238, -1711.7503662, 1790.4840088, -3641.2851562, 3644.2844238

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.0866390, upper bound: 24054.3437177
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8636702, upper bound: 24054.8592543
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15398.4541016, 16074.5263672, -11083.4912109, 11618.5419922, -27016.9960938, 27158.0175781
1: -1633.0220947, 1184.1640625, -1179.4667969, 856.3792114, -2489.4013672, 2363.6308594
2: -2679.7785645, 3044.4895020, -1926.0850830, 2207.3928223, -4887.1708984, 4970.5747070
3: -3100.4799805, 1925.1241455, -2233.7534180, 1388.1911621, -4488.6708984, 4158.8774414
4: -2346.4521484, 2445.1237793, -1692.0645752, 1770.7918701, -4117.2431641, 4137.1879883

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.0797000, upper bound: 24054.3275916
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8541896, upper bound: 24054.8541896
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12136.8769531, 12689.2626953, -11999.4648438, 12608.0654297, -24744.9375000, 24688.7265625
1: -1289.0649414, 937.8256836, -1279.4571533, 923.1782227, -2212.2431641, 2217.2827148
2: -2109.0053711, 2408.5756836, -2094.9565430, 2400.8222656, -4509.8276367, 4503.5322266
3: -2443.0090332, 1518.7714844, -2438.5881348, 1506.7264404, -3949.7353516, 3957.3593750
4: -1850.8012695, 1932.5344238, -1839.7347412, 1923.4959717, -3774.2973633, 3772.2690430

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15398.4541016, 16074.5263672, -11911.5292969, 12514.5781250, -27913.0312500, 27986.0546875
1: -1633.0220947, 1184.1640625, -1270.1448975, 916.3589478, -2549.3811035, 2454.3090820
2: -2679.7785645, 3044.4895020, -2079.6943359, 2383.4157715, -5063.1933594, 5124.1835938
3: -3100.4799805, 1925.1241455, -2421.3955078, 1495.6677246, -4596.1464844, 4346.5195312
4: -2346.4521484, 2445.1237793, -1826.5513916, 1909.4348145, -4255.8867188, 4271.6752930

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12394.0898438, 13183.2060547, -12647.7050781, 13432.7382812, -25826.8281250, 25830.9101562
1: -1331.2331543, 952.2580566, -1356.6655273, 971.9248657, -2303.1579590, 2308.9235840
2: -2159.0349121, 2496.6784668, -2202.4672852, 2544.9704590, -4704.0039062, 4699.1455078
3: -2501.4172363, 1564.9401855, -2550.6660156, 1595.0233154, -4096.4404297, 4115.6064453
4: -1887.6269531, 2007.5458984, -1926.1687012, 2046.4304199, -3934.0573730, 3933.7141113

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8515250
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8515250
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12231.0537109, 12855.1171875, -12647.7050781, 13432.7382812, -25663.7929688, 25502.8203125
1: -1304.3902588, 941.4983521, -1356.6655273, 971.9248657, -2276.3146973, 2298.1638184
2: -2133.5273438, 2447.8835449, -2202.4672852, 2544.9704590, -4678.4960938, 4650.3505859
3: -2483.5012207, 1535.7563477, -2550.6660156, 1595.0233154, -4078.5244141, 4086.4223633
4: -1874.3106689, 1961.4157715, -1926.1687012, 2046.4304199, -3920.7412109, 3887.5839844

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8530856
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8530856
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12303.6367188, 13037.1513672, -12316.4931641, 13004.1650391, -25307.8007812, 25353.6445312
1: -1318.0214844, 945.5207520, -1316.0555420, 946.8024292, -2264.8237305, 2261.5759277
2: -2143.5131836, 2471.8078613, -2145.6040039, 2468.3325195, -4611.8457031, 4617.4121094
3: -2483.7609863, 1549.9716797, -2485.8327637, 1548.1496582, -4031.9106445, 4035.8044434
4: -1875.4666748, 1986.7458496, -1878.3775635, 1983.2563477, -3858.7231445, 3865.1235352

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8686151
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8686340
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12027.0937500, 12602.1376953, -12316.4931641, 13004.1650391, -25031.2578125, 24918.6308594
1: -1280.1812744, 925.9521484, -1316.0555420, 946.8024292, -2226.9836426, 2242.0075684
2: -2099.7702637, 2402.6923828, -2145.6040039, 2468.3325195, -4568.1025391, 4548.2963867
3: -2445.1188965, 1507.5130615, -2485.8327637, 1548.1496582, -3993.2685547, 3993.3457031
4: -1846.0428467, 1924.4825439, -1878.3775635, 1983.2563477, -3829.2990723, 3802.8598633

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8712208
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8712208
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12685.2060547, 13470.8017578, -12328.6689453, 13095.1982422, -25780.4042969, 25799.4707031
1: -1361.0273438, 974.3457642, -1322.5061035, 946.9402466, -2307.9675293, 2296.8515625
2: -2212.0773926, 2549.9274902, -2148.4428711, 2480.6491699, -4692.7260742, 4698.3701172
3: -2560.9421387, 1600.5413818, -2488.1520996, 1555.1298828, -4116.0722656, 4088.6933594
4: -1933.1129150, 2050.9528809, -1878.4566650, 1994.7399902, -3927.8530273, 3929.4094238

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8505008
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8505043
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12692.7646484, 13304.9355469, -12328.6689453, 13095.1982422, -25787.9628906, 25633.6054688
1: -1351.6524658, 976.1735229, -1322.5061035, 946.9402466, -2298.5927734, 2298.6794434
2: -2217.0539551, 2532.6269531, -2148.4428711, 2480.6491699, -4697.7031250, 4681.0698242
3: -2580.1762695, 1591.7894287, -2488.1520996, 1555.1298828, -4135.3056641, 4079.9414062
4: -1947.4366455, 2029.7545166, -1878.4566650, 1994.7399902, -3942.1765137, 3908.2111816

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8525078
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8525078
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12614.2685547, 13360.4785156, -11994.7978516, 12680.6806641, -25294.9492188, 25355.2773438
1: -1351.0787354, 969.0847778, -1282.6116943, 921.5219116, -2272.6003418, 2251.6958008
2: -2200.0336914, 2531.1875000, -2090.8935547, 2405.7785645, -4605.8125000, 4622.0805664
3: -2547.6840820, 1589.2814941, -2422.1416016, 1509.1359863, -4056.8200684, 4011.4230957
4: -1923.8326416, 2035.0526123, -1829.3028564, 1933.4008789, -3857.2333984, 3864.3542480

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8649349
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8656209
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12521.0429688, 13103.9423828, -11994.7978516, 12680.6806641, -25201.7226562, 25098.7402344
1: -1332.3154297, 963.0625000, -1282.6116943, 921.5219116, -2253.8371582, 2245.6740723
2: -2188.7250977, 2496.5751953, -2090.8935547, 2405.7785645, -4594.5039062, 4587.4682617
3: -2548.3188477, 1569.2656250, -2422.1416016, 1509.1359863, -4057.4548340, 3991.4072266
4: -1923.6353760, 2000.1870117, -1829.3028564, 1933.4008789, -3857.0361328, 3829.4892578

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8683528
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8689676
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12477.4140625, 13146.9648438, -11348.4277344, 11886.2050781, -24363.6191406, 24495.3925781
1: -1332.0552979, 960.1785889, -1206.7696533, 876.6854858, -2208.7407227, 2166.9482422
2: -2174.8342285, 2499.3044434, -1971.9752197, 2257.0834961, -4431.9179688, 4471.2797852
3: -2527.9677734, 1568.3092041, -2285.8261719, 1420.5432129, -3948.5109863, 3854.1350098
4: -1907.9205322, 2004.6268311, -1730.9201660, 1811.2072754, -3719.1279297, 3735.5468750

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8809428, upper bound: 24054.8847351
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8809428, upper bound: 24054.8847351
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12955.6484375, 13617.0253906, -11244.6523438, 11773.3369141, -24728.9843750, 24861.6757812
1: -1381.2246094, 996.1814575, -1195.3750000, 868.5241699, -2249.7487793, 2191.5563965
2: -2258.0585938, 2587.6340332, -1953.4661865, 2235.8156738, -4493.8725586, 4541.1000977
3: -2624.5136719, 1626.0430908, -2264.3249512, 1407.3770752, -4031.8906250, 3890.3681641
4: -1980.5469971, 2075.6315918, -1715.1188965, 1793.9283447, -3774.4750977, 3790.7497559

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8807049, upper bound: 24054.8829725
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8807049, upper bound: 24054.8830671
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12477.4140625, 13146.9648438, -12344.5390625, 12968.4492188, -25445.8632812, 25491.5039062
1: -1332.0552979, 960.1785889, -1315.9667969, 950.3051147, -2282.3603516, 2276.1452637
2: -2174.8342285, 2499.3044434, -2152.8566895, 2469.0915527, -4643.9257812, 4652.1611328
3: -2527.9677734, 1568.3092041, -2505.0639648, 1549.6195068, -4077.5871582, 4073.3730469
4: -1907.9205322, 2004.6268311, -1891.0175781, 1978.4122314, -3886.3327637, 3895.6440430

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782688, upper bound: 24054.8808000
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782688, upper bound: 24054.8808000
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12955.6484375, 13617.0253906, -12160.5615234, 12767.3662109, -25723.0136719, 25777.5859375
1: -1381.2246094, 996.1814575, -1295.7279053, 935.6334839, -2316.8581543, 2291.9094238
2: -2258.0585938, 2587.6340332, -2120.1376953, 2430.2546387, -4688.3120117, 4707.7714844
3: -2624.5136719, 1626.0430908, -2466.2485352, 1526.1203613, -4150.6337891, 4092.2915039
4: -1980.5469971, 2075.6315918, -1861.3455811, 1947.3927002, -3927.9389648, 3936.9770508

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8777642, upper bound: 24054.8794360
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8777642, upper bound: 24054.8794548
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12953.5371094, 13613.7216797, -11211.5410156, 11747.8779297, -24701.4140625, 24825.2597656
1: -1381.0910645, 996.0867920, -1192.8001709, 866.2496338, -2247.3405762, 2188.8862305
2: -2260.8903809, 2587.4282227, -1948.4017334, 2232.1435547, -4493.0336914, 4535.8295898
3: -2627.7185059, 1626.1950684, -2259.6298828, 1403.7999268, -4031.5185547, 3885.8249512
4: -1983.1748047, 2075.6860352, -1711.7503662, 1790.4840088, -3773.6586914, 3787.4365234

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -16646.2265625, 17499.5625000, -11083.4912109, 11618.5419922, -28264.7695312, 28583.0546875
1: -1775.2087402, 1276.8702393, -1179.4667969, 856.3792114, -2631.5878906, 2456.3369141
2: -2900.0117188, 3316.8974609, -1926.0850830, 2207.3928223, -5107.4042969, 5242.9824219
3: -3364.0153809, 2089.8000488, -2233.7534180, 1388.1911621, -4752.2065430, 4323.5527344
4: -2539.3923340, 2662.9357910, -1692.0645752, 1770.7918701, -4310.1840820, 4354.9985352

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12953.5371094, 13613.7216797, -11999.4648438, 12608.0654297, -25561.6015625, 25613.1875000
1: -1381.0910645, 996.0867920, -1279.4571533, 923.1782227, -2304.2692871, 2275.5434570
2: -2260.8903809, 2587.4282227, -2094.9565430, 2400.8222656, -4661.7128906, 4682.3842773
3: -2627.7185059, 1626.1950684, -2438.5881348, 1506.7264404, -4134.4448242, 4064.7829590
4: -1983.1748047, 2075.6860352, -1839.7347412, 1923.4959717, -3906.6708984, 3915.4208984

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16646.2265625, 17499.5625000, -11911.5292969, 12514.5781250, -29160.8046875, 29411.0917969
1: -1775.2087402, 1276.8702393, -1270.1448975, 916.3589478, -2691.5676270, 2547.0151367
2: -2900.0117188, 3316.8974609, -2079.6943359, 2383.4157715, -5283.4272461, 5396.5917969
3: -3364.0153809, 2089.8000488, -2421.3955078, 1495.6677246, -4859.6831055, 4511.1953125
4: -2539.3923340, 2662.9357910, -1826.5513916, 1909.4348145, -4448.8271484, 4489.4858398

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 10.66 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8772841, upper bound: 24054.8734013
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8790035, upper bound: 24054.8751033
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8761779, upper bound: 24054.8731784
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8772804, upper bound: 24054.8748116
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8448151, upper bound: 24054.8514263
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8809156, upper bound: 24054.8784381
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8797919, upper bound: 24054.8785250
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8809156, upper bound: 24054.8784381
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8797919, upper bound: 24054.8785250
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8765868
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8778593
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8765868
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8757673, upper bound: 24054.8778593
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8631228, upper bound: 24054.8770984
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8835610, upper bound: 24054.8854778
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8630076, upper bound: 24054.8763662
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8836209, upper bound: 24054.8848433
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8811147, upper bound: 24054.8813063
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811353
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8815821, upper bound: 24054.8811746
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.0866390, upper bound: 24054.3437177
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8636702, upper bound: 24054.8592543
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.0797000, upper bound: 24054.3275916
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8541896, upper bound: 24054.8541896
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8532735, upper bound: 24054.8541650
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8515250
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8515250
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8530856
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8403718, upper bound: 24054.8530856
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8686151
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8686340
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8712208
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8655102, upper bound: 24054.8712208
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8505008
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8505043
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8525078
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8402301, upper bound: 24054.8525078
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8649349
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8656209
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8683528
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8653881, upper bound: 24054.8689676
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8809428, upper bound: 24054.8847351
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8809428, upper bound: 24054.8847351
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8807049, upper bound: 24054.8829725
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8807049, upper bound: 24054.8830671
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8782688, upper bound: 24054.8808000
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8782688, upper bound: 24054.8808000
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8777642, upper bound: 24054.8794360
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8777642, upper bound: 24054.8794548
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8541650, upper bound: 24054.8532735
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.66
Output dim: 0, lower bound: -24054.8508200, upper bound: 24054.8508200

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11087.0390625, 11534.9902344, -10925.1152344, 11465.8613281, -22552.9003906, 22460.1054688
1: -1172.5800781, 854.0237427, -1162.1340332, 840.9423828, -2013.5224609, 2016.1577148
2: -1932.7868652, 2195.3188477, -1900.7082520, 2180.7290039, -4113.5151367, 4096.0268555
3: -2247.6330566, 1382.1790771, -2207.6728516, 1369.2382812, -3616.8713379, 3589.8515625
4: -1700.2077637, 1760.4732666, -1670.3685303, 1748.9515381, -3449.1591797, 3430.8417969

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8772841, upper bound: 24054.8734013
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8745339, upper bound: 24054.8711304
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8745339, upper bound: 24054.8734013
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11230.3105469, 11684.6406250, -11258.3662109, 11777.8876953, -23008.1992188, 22943.0078125
1: -1187.5162354, 864.9824219, -1194.6145020, 866.7763672, -2054.2924805, 2059.5969238
2: -1957.0384521, 2222.7800293, -1956.1070557, 2242.3298340, -4199.3681641, 4178.8867188
3: -2274.4992676, 1399.8201904, -2271.9077148, 1407.4331055, -3681.9323730, 3671.7275391
4: -1721.0006104, 1783.0874023, -1719.7229004, 1799.1701660, -3520.1708984, 3502.8098145

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8790035, upper bound: 24054.8748702
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.06 + 415.46 = 420.52 seconds
